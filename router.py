"""
router.py — SonicSearch AI · Planificador (Manager)
=====================================================
Descarga el Router Prompt v4.1 desde el gist, construye la cadena
LangChain y expone get_router_chain() para ser importado por app.py.

Flujo:
    query (str)
        → ChatGroq (meta-llama/llama-4-scout-17b-16e-instruct)
        → JsonOutputParser
        → dict con claves: clap_positive, clap_negative,
                           lyric_positive {text, weight},
                           lyric_negative

Regla de Oro: todos los campos del JSON se generan en inglés,
independientemente del idioma de la query original.
"""

import os
import json
import requests

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_URL = (
    "https://gist.githubusercontent.com/AlexandraJMV/" 
    "da3e4b9afaacd9a2a6808967c8ae4966/raw/" 
    "cd01fd2ef05e71699ebe40bc6219c489d16909c9"
)
PROMPT_CACHE_PATH = "./api_prompt.txt"   # se guarda localmente tras la primera descarga

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ──────────────────────────────────────────────────────────────────────────────
# DESCARGA DEL PROMPT
# ──────────────────────────────────────────────────────────────────────────────

def _download_prompt() -> str:
    """
    Descarga el prompt desde el gist y lo cachea en disco.
    Si ya existe el archivo local, lo usa directamente (evita
    una petición de red en cada arranque del servidor).
    """
    if os.path.exists(PROMPT_CACHE_PATH):
        print(f"[ROUTER] Usando prompt cacheado en {PROMPT_CACHE_PATH}")
        with open(PROMPT_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()

    print(f"[ROUTER] Descargando prompt desde gist...")
    try:
        response = requests.get(PROMPT_URL, timeout=15)
        response.raise_for_status()
        prompt_text = response.text

        # Guardar en disco para próximos arranques
        with open(PROMPT_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        print(f"[ROUTER] Prompt descargado ({len(prompt_text)} chars). "
              f"Primeros 200: {prompt_text[:200]}...")
        return prompt_text

    except requests.RequestException as e:
        raise RuntimeError(
            f"No se pudo descargar el prompt desde el gist: {e}\n"
            f"Asegúrate de tener conexión a internet o coloca api_prompt.txt "
            f"manualmente en el directorio raíz del proyecto."
        )


# ──────────────────────────────────────────────────────────────────────────────
# FALLBACK — plan degradado si el LLM falla
# ──────────────────────────────────────────────────────────────────────────────

def _build_fallback_plan(query: str) -> dict:
    """
    Si el LLM o el parser fallan, devuelve un plan mínimo usando
    la query cruda. Permite que app.py continúe con la búsqueda.
    """
    return {
        "clap_positive":  query,
        "clap_negative":  "",
        "lyric_positive": {"text": query, "weight": 0.7},
        "lyric_negative": "",
        "_fallback":      True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA CADENA LANGCHAIN
# ──────────────────────────────────────────────────────────────────────────────

# Módulo-level cache: la cadena se construye una sola vez
_chain_cache = None

def get_router_chain():
    """
    Construye (o devuelve del caché) la cadena LangChain:
        prompt_template | ChatGroq | JsonOutputParser

    La cadena se invoca con:
        chain.invoke({"query": "..."})
    y devuelve un dict con las 4 claves del Router Prompt v4.1.
    """
    global _chain_cache
    if _chain_cache is not None:
        return _chain_cache

    # 1. Cargar el system prompt
    system_prompt = _download_prompt()

    # 2. Verificar API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY no encontrada. "
            "Ejecuta en CMD:  setx GROQ_API_KEY \"gsk_...\""
            " y reinicia la terminal."
        )

    # 3. Prompt template
    #    El system prompt del gist ya contiene las instrucciones completas.
    #    Solo inyectamos la query del usuario como human message.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

    # 4. Modelo
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        temperature=0,        # determinista: el router no debe ser creativo
        max_tokens=512,       # el JSON de salida es compacto
    )

    # 5. Parser con manejo de errores
    parser = JsonOutputParser()

    # 6. Cadena con fallback ante JSON malformado
    base_chain = prompt_template | llm | parser

    # Wrapper que captura fallos del parser y degrada graciosamente
    class RobustChain:
        def __init__(self, chain):
            self._chain = chain

        def invoke(self, inputs: dict) -> dict:
            query = inputs.get("query", "")
            try:
                result = self._chain.invoke(inputs)

                # Validar que el resultado tiene las claves mínimas esperadas
                _validate_plan(result)
                return result

            except Exception as e:
                print(f"[ROUTER] ⚠ Parser/LLM falló: {e}. Usando fallback.")
                return _build_fallback_plan(query)

    _chain_cache = RobustChain(base_chain)
    print(f"[ROUTER] ✅ Cadena lista — modelo: {GROQ_MODEL}")
    return _chain_cache


# ──────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN DEL PLAN
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"clap_positive", "clap_negative", "lyric_positive", "lyric_negative"}

def _validate_plan(plan: dict):
    """
    Verifica que el JSON devuelto por el LLM tiene la estructura
    esperada por el Router Prompt v4.1. Lanza ValueError si no.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"El plan no es un dict: {type(plan)}")

    missing = _REQUIRED_KEYS - set(plan.keys())
    if missing:
        raise ValueError(f"Claves faltantes en el plan: {missing}")

    lp = plan.get("lyric_positive")
    if isinstance(lp, dict):
        if "text" not in lp or "weight" not in lp:
            raise ValueError(f"lyric_positive debe tener 'text' y 'weight': {lp}")
        weight = float(lp["weight"])
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"lyric_positive.weight fuera de rango [0,1]: {weight}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST RÁPIDO (ejecutar directamente: python router.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Test directo de router.py")
    print("=" * 60)

    chain = get_router_chain()

    test_queries = [
        "canciones melancólicas con guitarra acústica",
        "algo con mucho bajo y ritmo para entrenar",
        "letra sobre soledad y lluvia",
    ]

    for q in test_queries:
        print(f"\n── Query: {q!r}")
        plan = chain.invoke({"query": q})
        print(json.dumps(plan, indent=2, ensure_ascii=False))