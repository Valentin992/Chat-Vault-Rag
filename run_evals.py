#!/usr/bin/env python3
"""
run_evals.py — Paso 7a del proyecto "Chat con mi Vault (RAG)".

La capa de evals: mide el sistema con un golden dataset (evals/golden.jsonl)
en vez de confiar en "se ve bien". Es lo que convierte el proyecto en un
sistema evaluado y no en un demo.

Qué mide:
  RETRIEVAL  recall@5 (¿la nota correcta está en el top-5?) y MRR (¿qué tan arriba?)
  GENERACIÓN groundedness (¿todo sale del contexto?), citas [n], cobertura de
             hechos clave, abstención en preguntas fuera del vault, latencia y costo

Modos:
    python run_evals.py --retrieval   # solo retrieval — casi gratis (solo embeddings)
    python run_evals.py               # completo — genera + juzga (~$0.40-0.60 por run)

El juez es Claude con structured outputs (Pydantic): devuelve un veredicto
JSON validado, no prosa que haya que parsear. Cada run se guarda en
evals/results/ para comparar runs en el tiempo.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic
from pydantic import BaseModel

from ask import MODEL, PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK, TOP_K, ask
from search import EMBED_MODEL, search

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

GOLDEN_PATH = Path(__file__).parent / "evals" / "golden.jsonl"
RESULTS_DIR = Path(__file__).parent / "evals" / "results"
JUDGE_MODEL = "claude-opus-4-8"

# El patrón de cita que exigimos en respuestas: [1], [2], ...
CITA_RE = re.compile(r"\[\d+\]")


class Veredicto(BaseModel):
    """Lo que el juez devuelve — JSON validado, no prosa."""

    se_abstuvo: bool          # ¿dijo "eso no está en tus notas"?
    usa_solo_contexto: bool   # groundedness: ¿toda afirmación está respaldada por el contexto?
    hechos_cubiertos: list[int]  # índices (base 0) de los hechos clave presentes en la respuesta
    comentario: str           # una frase de diagnóstico


JUDGE_SYSTEM = """Eres un juez estricto de evaluación de un sistema RAG.
Recibes: la PREGUNTA del usuario, el CONTEXTO que el sistema recuperó de las notas,
la RESPUESTA generada, y una lista numerada de HECHOS CLAVE esperados (puede estar vacía).

Evalúa exactamente esto:
- se_abstuvo: true si la respuesta dice que la información no está en las notas
  (en lugar de intentar responder).
- usa_solo_contexto: true si TODA afirmación factual de la respuesta está respaldada
  por el contexto. Si la respuesta agrega datos que no aparecen en el contexto, false.
  Si se abstuvo, true (no afirmó nada sin respaldo).
- hechos_cubiertos: los índices (base 0) de los hechos clave que la respuesta SÍ cubre.
  Acepta paráfrasis — importa el contenido, no las palabras exactas. Si la lista de
  hechos está vacía, devuelve [].
- comentario: una frase concreta de diagnóstico."""


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_retrieval(golden: list[dict]) -> dict:
    """recall@k y MRR sobre las preguntas respondibles. Solo cuesta embeddings."""
    answerable = [g for g in golden if g["type"] == "answerable"]
    rows, rr_sum, hits = [], 0.0, 0

    print(f"\n== RETRIEVAL (k={TOP_K}, {len(answerable)} preguntas) ==")
    for g in answerable:
        results = search(g["question"], k=TOP_K)
        # match por substring: evita fragilidad con acentos/rutas completas
        rank = next(
            (i for i, h in enumerate(results, 1)
             if any(exp in h["source"] for exp in g["expected_sources"])),
            None,
        )
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
        print(f"  {'✅' if rank else '❌'} {g['id']:16s} rank={rank}")
        rows.append({"id": g["id"], "rank": rank})

    metrics = {
        "recall_at_k": hits / len(answerable),
        "mrr": rr_sum / len(answerable),
        "hits": hits,
        "total": len(answerable),
    }
    print(f"  → recall@{TOP_K}: {hits}/{len(answerable)} ({metrics['recall_at_k']:.0%}) · MRR: {metrics['mrr']:.3f}")
    return {"metrics": metrics, "per_question": rows}


def judge(client: anthropic.Anthropic, g: dict, context: str, answer: str) -> tuple[Veredicto, float]:
    """Un veredicto estructurado por pregunta. Devuelve (veredicto, costo_usd)."""
    hechos = "\n".join(f"{i}. {f}" for i, f in enumerate(g["key_facts"])) or "(ninguno)"
    resp = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"PREGUNTA:\n{g['question']}\n\n"
                f"CONTEXTO RECUPERADO:\n{context}\n\n"
                f"RESPUESTA GENERADA:\n{answer}\n\n"
                f"HECHOS CLAVE ESPERADOS:\n{hechos}"
            ),
        }],
        output_format=Veredicto,
    )
    cost = (
        resp.usage.input_tokens * PRICE_IN_PER_MTOK
        + resp.usage.output_tokens * PRICE_OUT_PER_MTOK
    ) / 1_000_000
    return resp.parsed_output, cost


def eval_generation(golden: list[dict]) -> dict:
    """Loop completo + juez por pregunta. Esto sí cuesta (~$0.40-0.60 el run)."""
    client = anthropic.Anthropic()
    rows = []

    print(f"\n== GENERACIÓN ({len(golden)} preguntas, modelo {MODEL}, juez {JUDGE_MODEL}) ==")
    for g in golden:
        t0 = time.monotonic()
        result = ask(g["question"])
        latency = time.monotonic() - t0

        v, judge_cost = judge(client, g, result["context"], result["answer"])
        tiene_citas = bool(CITA_RE.search(result["answer"]))

        row = {
            "id": g["id"],
            "type": g["type"],
            "latency_s": round(latency, 1),
            "cost_ask_usd": result["usage"]["cost_usd"],
            "cost_judge_usd": round(judge_cost, 4),
            "tiene_citas": tiene_citas,
            "se_abstuvo": v.se_abstuvo,
            "usa_solo_contexto": v.usa_solo_contexto,
            "hechos_cubiertos": len(v.hechos_cubiertos),
            "hechos_totales": len(g["key_facts"]),
            "comentario": v.comentario,
        }
        rows.append(row)

        if g["type"] == "answerable":
            ok = v.usa_solo_contexto and not v.se_abstuvo
            detalle = f"hechos {row['hechos_cubiertos']}/{row['hechos_totales']} · citas {'sí' if tiene_citas else 'NO'}"
        else:
            ok = v.se_abstuvo
            detalle = "se abstuvo" if v.se_abstuvo else "ALUCINÓ (no se abstuvo)"
        print(f"  {'✅' if ok else '❌'} {g['id']:16s} {latency:5.1f}s  {detalle}")
        if not ok:
            print(f"     juez: {v.comentario}")

    ans = [r for r in rows if r["type"] == "answerable"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    metrics = {
        "groundedness": sum(r["usa_solo_contexto"] and not r["se_abstuvo"] for r in ans) / len(ans),
        "citas": sum(r["tiene_citas"] for r in ans) / len(ans),
        "cobertura_hechos": sum(r["hechos_cubiertos"] for r in ans) / sum(r["hechos_totales"] for r in ans),
        "abstencion": sum(r["se_abstuvo"] for r in oos) / len(oos) if oos else None,
        "latencia_media_s": round(sum(r["latency_s"] for r in rows) / len(rows), 1),
        "costo_medio_pregunta_usd": round(sum(r["cost_ask_usd"] for r in rows) / len(rows), 4),
        "costo_total_run_usd": round(sum(r["cost_ask_usd"] + r["cost_judge_usd"] for r in rows), 2),
    }
    print(
        f"  → groundedness: {metrics['groundedness']:.0%} · citas: {metrics['citas']:.0%}"
        f" · hechos: {metrics['cobertura_hechos']:.0%} · abstención: {metrics['abstencion']:.0%}"
    )
    print(
        f"  → latencia media: {metrics['latencia_media_s']}s"
        f" · costo/pregunta: ${metrics['costo_medio_pregunta_usd']}"
        f" · costo total del run (incl. juez): ${metrics['costo_total_run_usd']}"
    )
    return {"metrics": metrics, "per_question": rows}


def main():
    retrieval_only = "--retrieval" in sys.argv
    golden = load_golden()
    print(f"Golden dataset: {len(golden)} preguntas "
          f"({sum(g['type'] == 'answerable' for g in golden)} respondibles, "
          f"{sum(g['type'] == 'out_of_scope' for g in golden)} fuera del vault)")

    report = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "config": {"modelo": MODEL, "juez": JUDGE_MODEL, "embed": EMBED_MODEL, "k": TOP_K},
        "retrieval": eval_retrieval(golden),
    }
    if not retrieval_only:
        report["generacion"] = eval_generation(golden)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en: {out.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
