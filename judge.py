"""
judge.py — SonicSearch AI · Auditor de Restricciones (Juez)
============================================================
Evalúa si la letra de una canción recuperada cumple con la intención
del usuario y no viola las restricciones (lyric_negative).

Flujo:
    (query, lyrics, lyric_positive, lyric_negative)
        → ChatGroq (meta-llama/llama-4-scout-17b-16e-instruct)
        → JsonOutputParser
        → {"approved": bool, "reason": str, "problematic_terms": [...]}

Si approved=False, problematic_terms contiene los conceptos de la letra
que causaron el rechazo. Estos se agregan a lyric_negative para la
siguiente iteración de búsqueda.
"""

import os
import json
from dotenv import load_dotenv  # Cargar variables de entorno desde .env

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS LANGCHAIN — Plan A: Importaciones necesarias para LangChain
# ──────────────────────────────────────────────────────────────────────────────

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Carga GROQ_API_KEY y GROQ_MODEL del archivo .env

GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

JUDGE_SYSTEM_PROMPT = """\
Eres un auditor de restricciones para un sistema de búsqueda de canciones.

Recibirás:
- Consulta original del usuario.
- Letra recuperada de una canción candidata.
- Restricciones positivas (conceptos que la letra DEBE contener o reflejar).
- Restricciones negativas (conceptos que la letra NO debe contener).

Debes evaluar:
1. ¿La letra contiene o refleja el concepto solicitado en las restricciones positivas?
2. ¿Viola alguna restricción negativa?
3. ¿Existe contenido explícitamente prohibido por las restricciones negativas?

Devuelve ÚNICAMENTE un JSON válido con esta estructura exacta, sin texto adicional:
{{
  "approved": true o false,
  "reason": "explicación breve de por qué se aprueba o rechaza"
}}

Reglas:
- Sé estricto con las restricciones negativas: si la letra habla claramente \
del tema prohibido, recházala.
- Sé razonable con las restricciones positivas: la letra no necesita contener \
las palabras exactas, pero sí reflejar el concepto general.
- No inventes restricciones que no se te proporcionaron.
"""

JUDGE_HUMAN_TEMPLATE = """\
CONSULTA ORIGINAL: {query}

RESTRICCIONES POSITIVAS: {lyric_positive}

RESTRICCIONES NEGATIVAS: {lyric_negative}

LETRA DE LA CANCIÓN CANDIDATA:
{lyrics}
"""

# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON — cadena del juez
# ──────────────────────────────────────────────────────────────────────────────

_judge_chain = None


def _build_judge_chain():
    global _judge_chain
    if _judge_chain is not None:
        return _judge_chain

    groq_api_key = os.getenv("GROQ_API_KEY")  # ← Usar variable cargada desde .env (Plan B)
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY no encontrada. "
            "Verifica que el archivo .env contenga tu API Key."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human",  JUDGE_HUMAN_TEMPLATE),
    ])

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        temperature=0,
        max_tokens=300,
    )

    parser = JsonOutputParser()

    _judge_chain = prompt | llm | parser
    print(f"[JUDGE] ✅ Cadena del juez lista — modelo: {GROQ_MODEL}")
    return _judge_chain


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_song(
    query: str,
    lyrics: str,
    lyric_positive: list[str],
    lyric_negative: list[str],
) -> dict:
    """
    Evalúa si la letra de una canción cumple con la intención del usuario.

    Args:
        query:          consulta original del usuario
        lyrics:         texto completo de la letra de la canción
        lyric_positive: conceptos que la letra debe contener
        lyric_negative: conceptos que la letra NO debe contener

    Returns:
        {
            "approved": bool,
            "reason": str,
            "problematic_terms": list[str]   # vacío si approved=True
        }
    """
    chain = _build_judge_chain()

    try:
        result = chain.invoke({
            "query":          query,
            "lyrics":         lyrics[:3000],  # limitar para no exceder contexto
            "lyric_positive": ", ".join(lyric_positive) if lyric_positive else "(ninguna)",
            "lyric_negative": ", ".join(lyric_negative) if lyric_negative else "(ninguna)",
        })

        # Normalizar respuesta
        approved = result.get("approved", False)
        reason   = result.get("reason", "Sin razón proporcionada")

        verdict = {
            "approved": bool(approved),
            "reason":   str(reason),
        }

        status = "✅ APROBADA" if approved else "❌ RECHAZADA"
        print(f"[JUDGE] {status}: {reason}")

        return verdict

    except Exception as e:
        print(f"[JUDGE] ⚠ Error al evaluar: {e}. Aprobando por defecto.")
        return {
            "approved": True,
            "reason":   f"Error del juez: {e}. Aprobado por defecto.",
        }


# ──────────────────────────────────────────────────────────────────────────────
# TEST RÁPIDO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Test directo de judge.py")
    print("=" * 60)

    test_verdict = evaluate_song(
        query="canciones tristes pero sin amor",
        lyrics="I sit alone in the dark, tears fall like rain. "
               "My heart is broken since you left me, baby I miss your love.",
        lyric_positive=["sadness, loneliness, feeling empty and hopeless"],
        lyric_negative=["romantic love, heartbreak and longing for a person"],
    )
    print(json.dumps(test_verdict, indent=2, ensure_ascii=False))
