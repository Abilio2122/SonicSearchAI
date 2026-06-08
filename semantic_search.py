"""
semantic_search.py — SonicSearch AI · Worker Lírico
=====================================================
Búsqueda semántica sobre el índice FAISS de letras (Nomic embeddings).
Implementa el motor ortogonal: embedding positivo − negativos, renormalizado.
Aplica multi-query + RRF interno antes de devolver resultados a app.py.

Interfaz pública:
    search_lyrics(positive: list[str], negative: list[str], top_k: int)
        → [(song_id, rrf_score), ...]
"""

import os
import json
import numpy as np
import pandas as pd
import faiss

from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = "./output"
NOMIC_MODEL  = "nomic-ai/nomic-embed-text-v1.5"
RRF_K        = 60

# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON — carga pesada una sola vez
# ──────────────────────────────────────────────────────────────────────────────

_resources = None

def _load_resources() -> dict:
    global _resources
    if _resources is not None:
        return _resources

    print("[SEMANTIC] Cargando modelo Nomic...")
    embedder = SentenceTransformer(NOMIC_MODEL, trust_remote_code=True)

    print("[SEMANTIC] Cargando índice FAISS de letras...")
    index = faiss.read_index(os.path.join(OUTPUT_DIR, "faiss_lyrics.index"))
    print(f"  → {index.ntotal} vectores indexados")

    print("[SEMANTIC] Cargando datos alineados...")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "datos_alineados.csv"))
    df["song_id"] = df["song_id"].astype(int)

    print("[SEMANTIC] Cargando mapa de song_ids...")
    with open(os.path.join(OUTPUT_DIR, "song_id_map.json"), "r") as f:
        song_id_map = json.load(f)   # {"0": 213, "1": 45, ...}

    _resources = {
        "embedder":    embedder,
        "index":       index,
        "df":          df,
        "song_id_map": song_id_map,
    }
    print("[SEMANTIC] ✅ Worker lírico listo")
    return _resources


# ──────────────────────────────────────────────────────────────────────────────
# MOTOR ORTOGONAL
# ──────────────────────────────────────────────────────────────────────────────

def _build_query_vector(
    positive_text:  str,
    negative_texts: list[str],
    embedder:       SentenceTransformer,
) -> np.ndarray:
    vec = embedder.encode(
        f"search_query: {positive_text}",
        normalize_embeddings=True,
    ).astype(np.float32)

    for neg_text in negative_texts:
        neg_vec = embedder.encode(
            f"search_query: {neg_text}",
            normalize_embeddings=True,
        ).astype(np.float32)
        vec = vec - neg_vec

    # ── Truncar a 256-dim (Matryoshka) y renormalizar ──
    vec = vec[:256]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec

# ──────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA MULTI-QUERY RAW
# ──────────────────────────────────────────────────────────────────────────────

def _multi_query_search_raw(
    lyric_queries:   list[str],
    lyric_negatives: list[str],
    top_k:           int,
    resources:       dict,
) -> list[dict]:
    """
    Ejecuta una búsqueda FAISS por cada query positiva y acumula
    los resultados crudos con su origen para el RRF posterior.
    """
    embedder    = resources["embedder"]
    index       = resources["index"]
    df          = resources["df"]
    song_id_map = resources["song_id_map"]

    all_raw = []

    for i, query_text in enumerate(lyric_queries):
        q_vec = _build_query_vector(query_text, lyric_negatives, embedder)
        
        print(f"[SEMANTIC DEBUG] q_vec.shape={q_vec.shape}  index.d={index.d}")
        scores, indices = index.search(q_vec[np.newaxis, :], top_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            # Traducir posición FAISS → song_id
            song_id = int(song_id_map.get(str(idx), -1))
            if song_id == -1:
                continue

            all_raw.append({
                "song_id":      song_id,
                "score":        float(score),
                "query_origin": f"lyric_{i + 1}",
            })

    return all_raw


# ──────────────────────────────────────────────────────────────────────────────
# RRF INTERNO (entre múltiples queries de la misma rama)
# ──────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    all_raw: list[dict],
    k:       int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fusiona los resultados de múltiples queries líricas en un único ranking.
    Devuelve [(song_id, rrf_score), ...] ordenado descendente.
    """
    # Agrupar por query de origen
    by_query: dict[str, list[dict]] = defaultdict(list)
    for item in all_raw:
        by_query[item["query_origin"]].append(item)

    # Ordenar cada grupo por score coseno descendente
    for q in by_query:
        by_query[q].sort(key=lambda x: x["score"], reverse=True)

    # Acumular score RRF por song_id
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked_list in by_query.values():
        for rank, item in enumerate(ranked_list, start=1):
            rrf_scores[item["song_id"]] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

def search_lyrics(
    positive: list[str],
    negative: list[str],
    top_k:    int = 10,
) -> list[tuple[int, float]]:
    """
    Punto de entrada para app.py.

    Args:
        positive: lista de queries líricas positivas (del router)
        negative: lista de conceptos a sustraer ortogonalmente
        top_k:    candidatos por query antes de RRF

    Returns:
        [(song_id, rrf_score), ...] ordenado por relevancia descendente
    """
    if not positive:
        return []

    resources = _load_resources()

    # Filtrar strings vacíos
    positive = [q.strip() for q in positive if len(q.strip()) > 5]
    negative = [n.strip() for n in negative if len(n.strip()) > 5]

    if not positive:
        return []

    print(f"[SEMANTIC] Buscando: {len(positive)} queries positivas, "
          f"{len(negative)} negativas, top_k={top_k}")

    all_raw = _multi_query_search_raw(
        lyric_queries=positive,
        lyric_negatives=negative,
        top_k=top_k * 3,   # margen amplio antes del RRF
        resources=resources,
    )

    fused = _reciprocal_rank_fusion(all_raw)
    print(f"[SEMANTIC] → {len(fused)} candidatos tras RRF interno")

    return fused[:top_k]