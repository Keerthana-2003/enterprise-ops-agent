"""Evaluate the agent on a versioned task dataset and report quality, reliability, latency and cost.

Usage:
  python -m evals.run_evals                                  # OpenAI (needs OPENAI_API_KEY)
  python -m evals.run_evals --judge-model gpt-4o             # stronger judge model
  OPENAI_BASE_URL=http://localhost:11434/v1 AGENT_MODEL=llama3.1 python -m evals.run_evals --no-judge

Two scoring layers:
  1. Deterministic checks: required facts present, expected action proposed, no unwarranted actions.
  2. LLM-as-a-judge rubric (correctness, grounding, policy compliance, safety, actionability).

Tasks run concurrently (bounded by --concurrency). Results go to evals/results/<model>.json with the
dataset version (a content hash of tasks.jsonl) so results stay comparable as the dataset changes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path

from agent.core import Agent
from agent.providers import make_llm
from agent.seed import DATA, seed
from agent.telemetry import configure_otel

from .judge import judge

HERE = Path(__file__).parent


def answer_text(run: dict) -> str:
    final = run.get("final") or {}
    parts = [run.get("answer") or "", *final.get("facts_used", [])]
    return " ".join(parts).lower()


def grade(task: dict, run: dict) -> tuple[bool, str | None]:
    """Deterministic check. Returns (passed, failure_category)."""
    if run["status"] != "completed":
        return False, run["status"]
    text = answer_text(run)
    for group in task["expect_all"]:
        if not any(alt.lower() in text for alt in group):
            return False, "wrong_or_missing_facts"
    proposed = {a["action_type"] for a in run["proposed_actions"]}
    expected = task.get("expect_action")
    if expected and expected not in proposed:
        return False, "missing_expected_action"
    if not expected and proposed - {"no_action"}:
        return False, "unwarranted_action"
    return True, None


def load_dataset(path: Path) -> tuple[list[dict], str]:
    raw = path.read_text()
    tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task ids in dataset")
    return tasks, hashlib.sha256(raw.encode()).hexdigest()[:12]


async def evaluate(args: argparse.Namespace) -> dict:
    if not (DATA / "ontology.json").exists():
        seed()
    configure_otel("enterprise-ops-agent-evals")
    model = os.getenv("BEDROCK_MODEL_ID") if os.getenv("LLM_PROVIDER") == "bedrock" else os.getenv("AGENT_MODEL", "gpt-4o-mini")
    if args.runtime == "langgraph":
        from agent.graph import LangGraphAgent

        agent = LangGraphAgent(make_llm(model), DATA, max_steps=args.max_steps)
    else:
        agent = Agent(make_llm(model), DATA, max_steps=args.max_steps)
    judge_llm = None if args.no_judge else make_llm(args.judge_model)
    tasks, version = load_dataset(Path(args.tasks))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(task: dict) -> dict:
        async with sem:
            run = (await agent.arun(task["task"])).to_dict()
            passed, failure = grade(task, run)
            score = await judge(judge_llm, task["task"], run) if judge_llm else None
        tool_calls = sum(1 for s in run["steps"] if s["kind"] in ("tool", "guard") and s["name"] != "final_answer")
        row = {"id": task["id"], "passed": passed, "failure": failure, "status": run["status"],
               "structured": run["structured"], "tool_calls": tool_calls, "tool_errors": run["tool_errors"],
               "max_parallel_tool_calls": run["max_parallel_tool_calls"], "latency_ms": run["latency_ms"],
               "tokens": run["total_tokens"], "cached_tokens": run["cached_tokens"],
               "prompt_tokens": run["prompt_tokens"], "judge": score.model_dump() if score else None,
               "judge_mean": score.mean if score else None, "answer": run["answer"],
               "actions": [a["action_type"] for a in run["proposed_actions"]]}
        llm_errors = [s["error"] for s in run["steps"] if s["kind"] == "error"]
        row["llm_error"] = llm_errors[0] if llm_errors else None
        print(f"{'PASS' if passed else 'FAIL':4}  {task['id']:28} calls={tool_calls:2}  errors={run['tool_errors']}  "
              f"{run['latency_ms']/1000:5.1f}s  judge={row['judge_mean'] or '-'}  {failure or ''}")
        if row["llm_error"]:
            print(f"      -> {row['llm_error']}")
        return row

    rows = await asyncio.gather(*(one(t) for t in tasks))
    statuses = Counter(r["status"] for r in rows)
    failures = Counter(r["failure"] for r in rows if r["failure"])
    judged = [r for r in rows if r["judge"]]
    total_calls = sum(r["tool_calls"] for r in rows) or 1
    total_prompt = sum(r["prompt_tokens"] for r in rows) or 1
    latencies = sorted(r["latency_ms"] for r in rows)
    summary = {
        "model": model,
        "runtime": args.runtime,
        "dataset_version": version,
        "tasks": len(rows),
        "task_success_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "structured_output_rate": round(sum(r["structured"] for r in rows) / len(rows), 3),
        "judge_model": None if args.no_judge else args.judge_model,
        "judge_mean_score": round(statistics.mean(r["judge_mean"] for r in judged), 2) if judged else None,
        "judge_by_dimension": {k: round(statistics.mean(r["judge"][k] for r in judged), 2)
                               for k in ("correctness", "grounding", "policy_compliance", "safety", "actionability")}
                              if judged else None,
        "status_breakdown": dict(statuses),
        "failure_categories": dict(failures),
        "tool_error_rate": round(sum(r["tool_errors"] for r in rows) / total_calls, 3),
        "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2),
        "runs_with_parallel_tool_calls": sum(r["max_parallel_tool_calls"] > 1 for r in rows),
        "avg_latency_s": round(statistics.mean(latencies) / 1000, 2),
        "p95_latency_s": round(latencies[max(0, int(len(latencies) * 0.95) - 1)] / 1000, 2),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "prompt_cache_hit_rate": round(sum(r["cached_tokens"] for r in rows) / total_prompt, 3),
    }
    return {"summary": summary, "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(HERE / "tasks.jsonl"))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--runtime", choices=["loop", "langgraph"], default="loop")
    args = parser.parse_args()

    report = asyncio.run(evaluate(args))
    print("\n" + json.dumps(report["summary"], indent=2))
    name = f"{report['summary']['model']}_{report['summary']['runtime']}".replace("/", "_").replace(":", "_")
    out = HERE / "results" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
