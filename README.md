# free-model-aggregator-01: Min-Max Regret Minimizer Model Aggregator V2

Deterministic AI Model Aggregator submitted by `@sichigatsu` for the `free-model-aggregator-01` challenge ($500 pot).

## Strategy Overview
- **Solver**: `MinMaxRegretRouter` (Adversarial Regret Minimization)
- **Architecture**: Minimizes worst-case regret under upstream provider dropouts by guarding diverse fallback branches
- **Wire Support**: Dual-wire OpenAI Chat Completions (`/v1/chat/completions`) & Anthropic Messages (`/v1/messages`), supporting JSON and streaming SSE.
- **Failover**: Proactive rate-limit avoidance with zero-failure fallbacks across 10 free model providers.
