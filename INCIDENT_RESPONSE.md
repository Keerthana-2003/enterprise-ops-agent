# Incident response runbook

Who: the on-call engineer for the agent service. Where: `ops-agent` namespace.
First five minutes for any page:

1. Acknowledge the alert and post in the incident channel: what fired, when, current impact.
2. Check whether a rollout is in progress: `kubectl -n ops-agent get deploy` (an `agent-canary` or a
   recently switched blue/green colour is the most common cause).
3. If a change went out in the last hour, **roll back first, debug second** (commands below).
4. Open the trace for a failing run (Langfuse or Datadog APM, service `enterprise-ops-agent`).
5. Update the channel every 30 minutes until resolved; write a blameless postmortem within 3 days.

## Rollback commands

```bash
# canary in progress: remove it, stable keeps serving
kubectl -n ops-agent delete deploy/agent-canary
kubectl -n ops-agent scale deploy/agent-stable --replicas=4
# stable already updated: return to the previous ReplicaSet
kubectl -n ops-agent rollout undo deploy/agent-stable
# blue/green: point the Service back at the previous colour
kubectl -n ops-agent patch svc/agent-bg -p '{"spec":{"selector":{"app":"agent-bg","color":"blue"}}}'
```

## AgentHighErrorRate
5xx above 2% of requests for 5 minutes.
- `kubectl -n ops-agent logs -l app=agent --since=15m | grep -i error | sort | uniq -c | sort -rn | head`
- 5xx only on canary pods (`track=canary`)? Roll back the canary.
- All pods: check LLM provider status and `AgentLLMErrors`; check the latest config change.

## AgentHighLatencyP95
p95 of /runs above 20 s for 10 minutes.
- In traces, compare LLM call time vs tool time. Slow LLM: provider degradation or larger prompts
  (check `agent_tokens_total`). Slow tools: database or downstream API.
- CPU saturated? `kubectl -n ops-agent top pods`; confirm the HPA is scaling
  (`kubectl -n ops-agent describe hpa agent-stable`).

## AgentLLMErrors
More than 10% of runs end in `llm_error`.
- Error text is in the run trace and logs: `401` key revoked or rotated (update the `agent-secrets`
  Secret, then `kubectl rollout restart`), `429 insufficient_quota` billing, `429 rate_limit` traffic
  spike, `5xx` provider outage.
- Mitigation for provider outages: switch provider via the ConfigMap (`LLM_PROVIDER=bedrock`) and
  restart.

## AgentPodsUnavailable
No available stable pods.
- `kubectl -n ops-agent describe pods -l app=agent` (image pull errors, failed probes, OOMKilled,
  unschedulable). Roll back if it started with a deploy; otherwise fix capacity or the probe.

## AgentCrashLooping
- `kubectl -n ops-agent logs <pod> --previous` for the crash reason.
- OOMKilled: raise the memory limit or find the leak; config error: compare the ConfigMap with the
  last known good version in Git.

## After the incident
Postmortem template: summary, impact (duration, failed requests), timeline, root cause, what went
well, what went poorly, action items with owners and dates. Add a test or alert so the same failure
is caught earlier next time.
