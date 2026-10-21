"""Progressive delivery for the agent on Kubernetes.

Canary with gradual dial-up (default)
  1. one-box: start 1 canary pod next to the stable pods (the Service routes to both, so the canary
     gets traffic in proportion to its share of pods)
  2. bake, then compare the canary's error rate and p95 latency with the stable baseline
  3. dial up step by step (e.g. 25% -> 50% -> 100% of pods), checking after each bake
  4. promote (stable takes the new image, canary removed) or roll back automatically

Blue/green
  deploy the idle colour, smoke-test it, switch the Service selector, keep the old colour for an
  instant rollback.

Health is read from each pod's /metrics through the Kubernetes API server proxy, so no extra
tooling is needed. The decision logic is pure functions, unit-tested in tests/test_rollout.py.

  python deploy/rollout.py canary --image ghcr.io/me/agent:v2
  python deploy/rollout.py bluegreen --image ghcr.io/me/agent:v2
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LINE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{(.*)\})?\s+([-+0-9.eEinfNa]+)$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


# --------------------------------------------------------------------------- metric parsing and health

def parse_prometheus(text: str) -> list[tuple[str, dict, float]]:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if m:
            labels = dict(_LABEL.findall(m.group(3) or ""))
            out.append((m.group(1), labels, float(m.group(4))))
    return out


@dataclass
class Health:
    requests: float
    errors: float
    p95_s: float | None

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


def summarize(samples: list[tuple[str, dict, float]], ignore_routes=("/health", "/metrics")) -> Health:
    """Error rate and p95 latency of application routes from parsed /metrics samples (summed over pods)."""
    req = err = 0.0
    buckets: dict[float, float] = {}
    for name, labels, value in samples:
        if labels.get("route") in ignore_routes:
            continue
        if name == "http_requests_total":
            req += value
            if labels.get("status", "").startswith("5"):
                err += value
        elif name == "http_request_duration_seconds_bucket":
            le = math.inf if labels["le"] == "+Inf" else float(labels["le"])
            buckets[le] = buckets.get(le, 0.0) + value
    return Health(req, err, histogram_quantile(0.95, buckets))


def histogram_quantile(q: float, buckets: dict[float, float]) -> float | None:
    """Prometheus-style quantile from cumulative buckets {le: count}, linear within a bucket."""
    if not buckets:
        return None
    items = sorted(buckets.items())
    total = items[-1][1]
    if total == 0:
        return None
    rank, prev_le, prev_c = q * total, 0.0, 0.0
    for le, c in items:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            return prev_le + (le - prev_le) * ((rank - prev_c) / (c - prev_c) if c > prev_c else 0)
        prev_le, prev_c = le, c
    return prev_le


@dataclass
class Thresholds:
    max_error_rate: float = 0.02          # absolute ceiling for the canary
    max_error_rate_delta: float = 0.01    # canary may not exceed baseline by more than this
    max_p95_ratio: float = 1.5            # canary p95 vs baseline p95
    min_requests: int = 20                # below this, judge on errors only


@dataclass
class Decision:
    healthy: bool
    reasons: list[str] = field(default_factory=list)


def evaluate(canary: Health, baseline: Health, t: Thresholds) -> Decision:
    reasons = []
    if canary.error_rate > t.max_error_rate:
        reasons.append(f"canary error rate {canary.error_rate:.2%} > {t.max_error_rate:.2%}")
    if canary.error_rate - baseline.error_rate > t.max_error_rate_delta:
        reasons.append(f"canary error rate {canary.error_rate:.2%} exceeds baseline {baseline.error_rate:.2%}")
    if canary.requests >= t.min_requests and canary.p95_s and baseline.p95_s:
        if canary.p95_s > t.max_p95_ratio * baseline.p95_s:
            reasons.append(f"canary p95 {canary.p95_s:.2f}s > {t.max_p95_ratio}x baseline {baseline.p95_s:.2f}s")
    return Decision(not reasons, reasons)


def dial_up_plan(total_pods: int, percents=(25, 50, 100)) -> list[tuple[int, int]]:
    """(canary_pods, stable_pods) per step, starting with a one-box canary. Pod share approximates
    traffic share because the Service load-balances across all ready pods."""
    steps = [(1, total_pods)]  # one-box: a single extra pod
    for p in percents:
        canary = max(1, round(total_pods * p / 100))
        step = (canary, max(total_pods - canary, 0))
        if step != steps[-1]:
            steps.append(step)
    return steps


# --------------------------------------------------------------------------- kubectl wrappers

def kubectl(*args: str, input_text: str | None = None, check: bool = True) -> str:
    r = subprocess.run(["kubectl", *args], input=input_text, text=True, capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def pod_metrics(ns: str, selector: str) -> list[tuple[str, dict, float]]:
    pods = json.loads(kubectl("get", "pods", "-n", ns, "-l", selector, "-o", "json"))["items"]
    samples = []
    for p in pods:
        if p["status"].get("phase") == "Running":
            name = p["metadata"]["name"]
            samples += parse_prometheus(kubectl("get", "--raw", f"/api/v1/namespaces/{ns}/pods/{name}:8000/proxy/metrics"))
    return samples


def render(path: Path, **values: str) -> str:
    text = path.read_text()
    for k, v in values.items():
        text = text.replace("${" + k + "}", v)
    return text


def canary(args) -> int:
    ns, t = args.namespace, Thresholds()
    total = int(kubectl("get", "deploy/agent-stable", "-n", ns, "-o", "jsonpath={.spec.replicas}") or 1)
    plan = dial_up_plan(total)
    kubectl("apply", "-f", "-", input_text=render(ROOT / "k8s" / "canary" / "deployment-canary.yaml", IMAGE=args.image))
    for i, (c, s) in enumerate(plan):
        label = "one-box" if i == 0 else f"{round(100 * c / max(c + s, 1))}% of pods"
        print(f"step {i + 1}/{len(plan)} ({label}): canary={c} stable={s}")
        kubectl("scale", "deploy/agent-canary", "-n", ns, f"--replicas={c}")
        if i > 0:
            kubectl("scale", "deploy/agent-stable", "-n", ns, f"--replicas={max(s, 1) if c < total else 1}")
        kubectl("rollout", "status", "deploy/agent-canary", "-n", ns, f"--timeout={args.timeout}s")
        time.sleep(args.bake)
        d = evaluate(summarize(pod_metrics(ns, "app=agent,track=canary")),
                     summarize(pod_metrics(ns, "app=agent,track=stable")), t)
        print("  healthy" if d.healthy else "  UNHEALTHY: " + "; ".join(d.reasons))
        if not d.healthy:
            print("rolling back: removing canary, restoring stable")
            kubectl("scale", "deploy/agent-stable", "-n", ns, f"--replicas={total}")
            kubectl("delete", "deploy/agent-canary", "-n", ns, "--ignore-not-found")
            return 1
    print("promoting: stable takes the canary image")
    kubectl("set", "image", "deploy/agent-stable", f"api={args.image}", "-n", ns)
    kubectl("scale", "deploy/agent-stable", "-n", ns, f"--replicas={total}")
    kubectl("rollout", "status", "deploy/agent-stable", "-n", ns, f"--timeout={args.timeout}s")
    kubectl("delete", "deploy/agent-canary", "-n", ns, "--ignore-not-found")
    return 0


def bluegreen(args) -> int:
    ns = args.namespace
    live = kubectl("get", "svc/agent-bg", "-n", ns, "-o", "jsonpath={.spec.selector.color}", check=False).strip() or "blue"
    idle = "green" if live == "blue" else "blue"
    print(f"live={live}, deploying {idle} with {args.image}")
    kubectl("apply", "-f", "-", input_text=render(ROOT / "k8s" / "bluegreen" / "deployment.yaml", COLOR=idle, IMAGE=args.image))
    kubectl("rollout", "status", f"deploy/agent-{idle}", "-n", ns, f"--timeout={args.timeout}s")
    pods = json.loads(kubectl("get", "pods", "-n", ns, "-l", f"app=agent-bg,color={idle}", "-o", "json"))["items"]
    for p in pods:  # smoke test every new pod before it takes traffic
        body = kubectl("get", "--raw", f"/api/v1/namespaces/{ns}/pods/{p['metadata']['name']}:8000/proxy/health")
        if json.loads(body).get("status") != "ok":
            print(f"smoke test failed on {p['metadata']['name']}; {live} stays live")
            return 1
    kubectl("patch", "svc/agent-bg", "-n", ns, "-p", json.dumps({"spec": {"selector": {"app": "agent-bg", "color": idle}}}))
    print(f"switched traffic to {idle}; roll back with: kubectl patch svc/agent-bg -n {ns} "
          f"-p '{{\"spec\":{{\"selector\":{{\"app\":\"agent-bg\",\"color\":\"{live}\"}}}}}}'")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=["canary", "bluegreen"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--namespace", default="ops-agent")
    ap.add_argument("--bake", type=int, default=120, help="seconds to observe each step")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    sys.exit(canary(args) if args.strategy == "canary" else bluegreen(args))


if __name__ == "__main__":
    main()
