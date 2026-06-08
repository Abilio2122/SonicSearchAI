""""
app.py — SonicSearch AI · Servidor Flask
=========================================
Orquesta el flujo Plan-and-Execute:
  1. Router (LLM Groq) → plan JSON
  2. Workers → búsqueda lírica + acústica
  3. Fusión RRF + pesos dinámicos
  4. Lookup en datos_alineados.csv

Formato real del router (listas, no strings):
  {
    "clap_positive":  ["texto 1", "texto 2"],
    "clap_negative":  [],
    "lyric_positive": ["texto 1", "texto 2"],
    "lyric_negative": []
  }

Pesos dinámicos inferidos desde la presencia de queries en cada rama:
  - Solo lírica presente  → lyric 0.9 / audio 0.1
  - Solo audio presente   → lyric 0.1 / audio 0.9
  - Ambas presentes       → lyric 0.6 / audio 0.4  (lírica priorizada por defecto)
  - Ninguna               → fallback query cruda, pesos 0.7 / 0.3
"""
import json
import os
import traceback
import numpy as np
import pandas as pd
import faiss

from flask import Flask, render_template, request, jsonify

# ── Imports lazy ──────────────────────────────────────────────────────────────
def _import_router():
    from router import get_router_chain
    return get_router_chain()

def _import_semantic_search():
    from semantic_search import search_lyrics
    return search_lyrics

def _import_clap_search():
    from clap_search import search_audio
    return search_audio

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "./output"
TOP_K       = 10
FINAL_TOP_K = 5
RRF_K       = 60

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SEARCH ENGINE — carga índices y metadata una sola vez
# ──────────────────────────────────────────────────────────────────────────────

class SearchEngine:
    def __init__(self):
        self.df_meta         = None
        self.song_id_map     = None   # posición FAISS (str) → song_id (int)
        self.idx_lyrics      = None
        self.idx_audio       = None
        self.ready           = False
        self.error           = None

    def load_indices(self):
        try:
            print("[ENGINE] Cargando datos alineados...")
            self.df_meta = pd.read_csv(os.path.join(OUTPUT_DIR, "datos_alineados.csv"))
            self.df_meta["song_id"] = self.df_meta["song_id"].astype(int)
            print(f"  → {len(self.df_meta)} canciones")

            print("[ENGINE] Cargando mapa de song_ids...")
            with open(os.path.join(OUTPUT_DIR, "song_id_map.json"), "r") as f:
                self.song_id_map = json.load(f)   # {"0": 213, "1": 45, ...}
            print(f"  → {len(self.song_id_map)} entradas")

            print("[ENGINE] Cargando índice FAISS de letras...")
            self.idx_lyrics = faiss.read_index(os.path.join(OUTPUT_DIR, "faiss_lyrics.index"))
            print(f"  → {self.idx_lyrics.ntotal} vectores (lyrics)")

            print("[ENGINE] Cargando índice FAISS de audio...")
            self.idx_audio = faiss.read_index(os.path.join(OUTPUT_DIR, "faiss_audio.index"))
            print(f"  → {self.idx_audio.ntotal} vectores (audio)")

            self.ready = True
            print("[ENGINE] ✅ Motor listo")

        except Exception as e:
            self.error = str(e)
            print(f"[ENGINE] ❌ Error: {e}")
            traceback.print_exc()

    def get_song_by_id(self, song_id: int) -> dict | None:
        row = self.df_meta[self.df_meta["song_id"] == song_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "song_id": int(r["song_id"]),
            "title":   str(r.get("title",  "—")),
            "artist":  str(r.get("artist", "—")),
            "tag":     str(r.get("tag",    "—")),
            "year":    str(r.get("year",   "—")),
        }


engine = SearchEngine()


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCIA DE PESOS desde el plan del router
# ──────────────────────────────────────────────────────────────────────────────

def _infer_weights(lyric_positive: list, clap_positive: list) -> tuple[float, float]:
    """
    Infiere pesos dinámicos según qué ramas tiene el router pobladas.

    Casos:
      Ambas presentes  → 0.6 lírica / 0.4 audio  (lírica priorizada)
      Solo lírica      → 0.9 / 0.1
      Solo audio       → 0.1 / 0.9
      Ninguna          → 0.7 / 0.3  (fallback balanceado)
    """
    has_lyric = bool(lyric_positive)
    has_audio = bool(clap_positive)

    if has_lyric and has_audio:
        return 0.6, 0.4
    elif has_lyric:
        return 0.9, 0.1
    elif has_audio:
        return 0.1, 0.9
    else:
        return 0.7, 0.3


# ──────────────────────────────────────────────────────────────────────────────
# FUSIÓN HÍBRIDA (Late Fusion — RRF + suma ponderada)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_scores(results: list[tuple[int, float]]) -> dict[int, float]:
    if not results:
        return {}
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return {sid: 1.0 for sid, _ in results}
    return {sid: (s - min_s) / (max_s - min_s) for sid, s in results}


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)


def compute_hybrid_score(
    lyrics_results: list[tuple[int, float]],
    audio_results:  list[tuple[int, float]],
    lyric_weight:   float = 0.6,
    audio_weight:   float = 0.4,
) -> list[tuple[int, float]]:
    norm_lyrics = _normalize_scores(lyrics_results)
    norm_audio  = _normalize_scores(audio_results)

    rank_lyrics = {sid: rank for rank, (sid, _) in enumerate(lyrics_results)}
    rank_audio  = {sid: rank for rank, (sid, _) in enumerate(audio_results)}

    all_ids = set(norm_lyrics) | set(norm_audio)
    fused = {}

    for sid in all_ids:
        base = lyric_weight * norm_lyrics.get(sid, 0.0) + \
               audio_weight * norm_audio.get(sid,  0.0)

        rrf = 0.0
        if sid in rank_lyrics:
            rrf += _rrf_score(rank_lyrics[sid])
        if sid in rank_audio:
            rrf += _rrf_score(rank_audio[sid])

        fused[sid] = base + 0.1 * rrf

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def get_top_k_results(query: str, k: int = FINAL_TOP_K) -> dict:
    if not engine.ready:
        return {"error": f"Motor no inicializado: {engine.error}"}

    # ── PLAN ──────────────────────────────────────────────────────────────────
    try:
        chain = _import_router()
        plan  = chain.invoke({"query": query})

        # Formato real: todas las claves son listas de strings
        clap_positive  = plan.get("clap_positive",  [])
        clap_negative  = plan.get("clap_negative",  [])
        lyric_positive = plan.get("lyric_positive", [])
        lyric_negative = plan.get("lyric_negative", [])

        # Normalizar: si alguna clave llegó como string, envolverla en lista
        if isinstance(clap_positive,  str): clap_positive  = [clap_positive]
        if isinstance(clap_negative,  str): clap_negative  = [clap_negative]
        if isinstance(lyric_positive, str): lyric_positive = [lyric_positive]
        if isinstance(lyric_negative, str): lyric_negative = [lyric_negative]

        # Inferir pesos desde presencia de queries
        lyric_weight, audio_weight = _infer_weights(lyric_positive, clap_positive)

        print(f"[PLAN] lyric={len(lyric_positive)}q  audio={len(clap_positive)}q  "
              f"pesos={lyric_weight}/{audio_weight}")

    except Exception as e:
        print(f"[PLAN] Router falló, usando query directa: {e}")
        clap_positive  = [query]
        clap_negative  = []
        lyric_positive = [query]
        lyric_negative = []
        lyric_weight   = 0.7
        audio_weight   = 0.3
        plan           = {"_fallback": str(e)}

    # ── EXEC: Rama Lírica ─────────────────────────────────────────────────────
    lyrics_results = []
    if lyric_positive:
        try:
            search_lyrics  = _import_semantic_search()
            lyrics_results = search_lyrics(
                positive=lyric_positive,   # lista de strings
                negative=lyric_negative,   # lista de strings
                top_k=TOP_K,
            )
            print(f"[EXEC] Lírica: {len(lyrics_results)} candidatos")
        except Exception as e:
            print(f"[EXEC] Búsqueda lírica falló: {e}")
            traceback.print_exc()

    # ── EXEC: Rama Acústica ───────────────────────────────────────────────────
    audio_results = []
    if clap_positive:
        try:
            search_audio_fn = _import_clap_search()
            audio_results   = search_audio_fn(
                positive=clap_positive,    # lista de strings
                negative=clap_negative,    # lista de strings
                top_k=TOP_K,
            )
            print(f"[EXEC] Audio: {len(audio_results)} candidatos")
        except Exception as e:
            print(f"[EXEC] Búsqueda acústica falló: {e}")

    # ── FUSE ──────────────────────────────────────────────────────────────────
    if not lyrics_results and not audio_results:
        return {
            "error": "Ambas ramas de búsqueda fallaron.",
            "plan":  plan,
        }

    fused = compute_hybrid_score(
        lyrics_results,
        audio_results,
        lyric_weight=lyric_weight,
        audio_weight=audio_weight,
    )

    # ── BUILD RESPONSE ────────────────────────────────────────────────────────
    results = []
    for song_id, score in fused[:k]:
        meta = engine.get_song_by_id(song_id)
        if meta:
            meta["score"]     = round(score, 4)
            meta["score_pct"] = round(score * 100, 1)
            results.append(meta)

    return {
        "query":        query,
        "plan":         plan,
        "results":      results,
        "lyric_weight": lyric_weight,
        "audio_weight": audio_weight,
        "n_lyrics":     len(lyrics_results),
        "n_audio":      len(audio_results),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS FLASK
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    body  = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "El campo 'query' es requerido."}), 400
    try:
        return jsonify(get_top_k_results(query, k=FINAL_TOP_K))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "ready":          engine.ready,
        "error":          engine.error,
        "songs":          len(engine.df_meta) if engine.df_meta is not None else 0,
        "lyrics_vectors": engine.idx_lyrics.ntotal if engine.idx_lyrics else 0,
        "audio_vectors":  engine.idx_audio.ntotal  if engine.idx_audio  else 0,
    })


# ──────────────────────────────────────────────────────────────────────────────
# ARRANQUE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine.load_indices()
    app.run(debug=True, port=5000)
