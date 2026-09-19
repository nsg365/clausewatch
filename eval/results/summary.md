# Evaluation results

Generator: `openai/gpt-oss-120b` (groq) | embeddings: `BAAI/bge-small-en-v1.5` | reranker: `BAAI/bge-reranker-base`

RAGAS judges - context_precision: `qwen/qwen3.8-27b`, faithfulness: `openai/gpt-oss-20b`

## End-to-end (RAGAS + deterministic checks)

| Config | Faithfulness | Answer relevancy | Context precision | Citation grounding | Abstention (unanswerable) | Routing acc. | p50 latency (s) | Tokens / question |
|---|---|---|---|---|---|---|---|---|
| full | 0.94 | 0.85 | 0.88 | 1.00 | 1.00 | 1.00 | 30.60 | 7036 |
| no_verifier | 0.91 | - | - | 0.94 | 1.00 | 1.00 | 17.60 | 4326 |

## Retrieval ablation (18 questions, no LLM)

| Query | Scope | Retriever | Evidence recall@6 | MRR | Hit@1 |
|---|---|---|---|---|---|
| router | scoped | Dense only | 0.89 | 0.78 | 0.67 |
| router | scoped | BM25 only | 1.00 | 0.97 | 0.94 |
| router | scoped | Hybrid (RRF) | 1.00 | 0.87 | 0.78 |
| router | scoped | Hybrid + cross-encoder rerank | 0.94 | 0.92 | 0.89 |
| router | scoped | Hybrid + clause boost + rerank | 0.94 | 0.92 | 0.89 |
| question | scoped | Dense only | 0.92 | 0.84 | 0.78 |
| question | scoped | BM25 only | 0.94 | 0.88 | 0.83 |
| question | scoped | Hybrid (RRF) | 1.00 | 0.86 | 0.78 |
| question | scoped | Hybrid + cross-encoder rerank | 0.92 | 0.84 | 0.78 |
| question | open | Dense only | 0.50 | 0.43 | 0.39 |
| question | open | BM25 only | 0.56 | 0.35 | 0.22 |
| question | open | Hybrid (RRF) | 0.61 | 0.48 | 0.44 |
| question | open | Hybrid + cross-encoder rerank | 0.61 | 0.50 | 0.44 |

Clause tagger vs CUAD expert labels (chunk level): recall 0.52, precision 0.18.


## Risk flagging on held-out contracts

| Contract | Flags | Rejected by verifier | Consistent with CUAD labels |
|---|---|---|---|
| ArmstrongFlooringInc - Intellectual Property Agreement | 1 | 2 | 1/1 |
| N2KINC - Sponsorship Agreement | 1 | 1 | 1/1 |
| WARNINGMANAGEMENTSERVICESINC - Endorsement Agreement | 3 | 0 | 1/2 |
