"""
clap_search.py — SonicSearch AI · Worker Acústico
==================================================
Búsqueda semántica sobre el índice FAISS de audio (CLAP embeddings).
Usa la rama texto→audio de CLAP: convierte descripciones textuales
al mismo espacio latente que los embeddings de audio del corpus.
Aplica el mismo motor ortogonal y RRF interno que el worker lírico.

Modelo: laion/clap-htsat-unfused  (mismo que generó los embeddings)

Interfaz pública:
    search_audio(positive: list[str], negative: list[str], top_k: int)
        → [(song_id, rrf_score), ...]
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
import torch

from collections import defaultdict
from transformers import ClapModel, ClapProcessor

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "./output"
CLAP_MODEL  = "laion/clap-htsat-unfused"
RRF_K       = 60
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON — carga pesada una sola vez
# ──────────────────────────────────────────────────────────────────────────────

_resources = None

def _load_resources() -> dict:
    global _resources
    if _resources is not None:
        return _resources

    print(f"[CLAP] Cargando modelo CLAP en {DEVICE}...")
    model = ClapModel.from_pretrained(CLAP_MODEL).to(DEVICE)
    model.eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    print("[CLAP] Modelo cargado")

    print("[CLAP] Cargando índice FAISS de audio...")
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "faiss_audio.index"))
    print(f"  → {index.ntotal} vectores indexados")

    print("[CLAP] Cargando datos alineados...")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "datos_alineados.csv"))
    df["song_id"] = df["song_id"].astype(int)

    print("[CLAP] Cargando mapa de song_ids...")
    with open(os.path.join(OUTPUT_DIR, "song_id_map.json"), "r") as f:
        song_id_map = json.load(f)   # {"0": 213, "1": 45, ...}

    _resources = {
        "model":       model,
        "processor":   processor,
        "index":       index,
        "df":          df,
        "song_id_map": song_id_map,
    }
    print("[CLAP] ✅ Worker acústico listo")
    return _resources


# ──────────────────────────────────────────────────────────────────────────────
# ENCODER DE TEXTO → ESPACIO CLAP
# ──────────────────────────────────────────────────────────────────────────────
def _encode_text(text: str, model: ClapModel, processor: ClapProcessor) -> np.ndarray:
    inputs = processor(text=text, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        output = model.get_text_features(**inputs)

    # Debug: mostrar tipo y atributos disponibles
    print(f"[CLAP DEBUG] type(output): {type(output)}")

    # Extraer tensor según lo que retorne
    if isinstance(output, torch.Tensor):
        vec_tensor = output
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        vec_tensor = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        vec_tensor = output.last_hidden_state[:, 0, :]  # CLS token
    else:
        # Último recurso: primer elemento si es iterable
        vec_tensor = output[0]
        print(f"[CLAP DEBUG] Usando output[0], shape: {vec_tensor.shape}")

    print(f"[CLAP DEBUG] vec_tensor.shape: {vec_tensor.shape}")

    vec = vec_tensor.squeeze().cpu().float().numpy()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# MOTOR ORTOGONAL CLAP
# ──────────────────────────────────────────────────────────────────────────────

def _build_query_vector(
    positive_text:  str,
    negative_texts: list[str],
    model:          ClapModel,
    processor:      ClapProcessor,
) -> np.ndarray:
    """
    Vector de consulta con rechazo ortogonal en el espacio CLAP:
        vec = encode(positive) − Σ encode(negative_i)
    Renormalizado a norma unitaria.
    """
    vec = _encode_text(positive_text, model, processor)

    for neg_text in negative_texts:
        neg_vec = _encode_text(neg_text, model, processor)
        vec = vec - neg_vec

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec


# ──────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA MULTI-QUERY RAW
# ──────────────────────────────────────────────────────────────────────────────

def _multi_query_search_raw(
    clap_queries:   list[str],
    clap_negatives: list[str],
    top_k:          int,
    resources:      dict,
) -> list[dict]:
    """
    Ejecuta una búsqueda FAISS por cada query acústica positiva.
    """
    model       = resources["model"]
    processor   = resources["processor"]
    index       = resources["index"]
    song_id_map = resources["song_id_map"]

    all_raw = []

    for i, query_text in enumerate(clap_queries):
        q_vec = _build_query_vector(query_text, clap_negatives, model, processor)
        scores, indices = index.search(q_vec[np.newaxis, :], top_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            song_id = int(song_id_map.get(str(idx), -1))
            if song_id == -1:
                continue

            all_raw.append({
                "song_id":      song_id,
                "score":        float(score),
                "query_origin": f"clap_{i + 1}",
            })

    return all_raw


# ──────────────────────────────────────────────────────────────────────────────
# RRF INTERNO
# ──────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    all_raw: list[dict],
    k:       int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fusiona resultados de múltiples queries acústicas en un único ranking.
    Devuelve [(song_id, rrf_score), ...] ordenado descendente.
    """
    by_query: dict[str, list[dict]] = defaultdict(list)
    for item in all_raw:
        by_query[item["query_origin"]].append(item)

    for q in by_query:
        by_query[q].sort(key=lambda x: x["score"], reverse=True)

    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked_list in by_query.values():
        for rank, item in enumerate(ranked_list, start=1):
            rrf_scores[item["song_id"]] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

def search_audio(
    positive: list[str],
    negative: list[str],
    top_k:    int = 10,
) -> list[tuple[int, float]]:
    """
    Punto de entrada para app.py.

    Args:
        positive: lista de descripciones acústicas positivas (del router)
        negative: lista de conceptos acústicos a sustraer
        top_k:    candidatos por query antes de RRF

    Returns:
        [(song_id, rrf_score), ...] ordenado por relevancia descendente
    """
    if not positive:
        return []

    resources = _load_resources()

    positive = [q.strip() for q in positive if len(q.strip()) > 5]
    negative = [n.strip() for n in negative if len(n.strip()) > 5]

    if not positive:
        return []

    print(f"[CLAP] Buscando: {len(positive)} queries positivas, "
          f"{len(negative)} negativas, top_k={top_k}")

    all_raw = _multi_query_search_raw(
        clap_queries=positive,
        clap_negatives=negative,
        top_k=top_k * 3,
        resources=resources,
    )

    fused = _reciprocal_rank_fusion(all_raw)
    print(f"[CLAP] → {len(fused)} candidatos tras RRF interno")

    return fused[:top_k]