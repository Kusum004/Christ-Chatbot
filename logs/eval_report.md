# Evaluation report

- LLM: `Qwen/Qwen2.5-0.5B-Instruct` on `cuda`; PEFT adapter for Model 3: `lora-helpdesk-v1`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, hybrid alpha=0.6, top_k=4
- Questions: 31 (26 answerable, 5 unanswerable/out-of-scope)
- Retrieval Hit@4: **1.00**, MRR: **0.97**

| system                         |   Keyword recall (answerable) |   Groundedness of answers |   Refusal accuracy (unanswerable) |   Wrong refusals (answerable) |   Avg latency ms |   p95 latency ms |   Avg output tokens |
|:-------------------------------|------------------------------:|--------------------------:|----------------------------------:|------------------------------:|-----------------:|-----------------:|--------------------:|
| Model 1 - LLM only             |                         0.173 |                     0.144 |                               0.2 |                         0.038 |         3649.1   |          7773.26 |             101.903 |
| Model 2 - RAG                  |                         0.904 |                     0.847 |                               0.4 |                         0     |         1696.33  |          4744.37 |              46.194 |
| Model 3 - RAG + LLMOps + PEFT  |                         0.712 |                     1     |                               0.8 |                         0.115 |          987.882 |          1825.89 |              26.194 |
| Ablation: Model 3 without PEFT |                         0.904 |                     0.882 |                               0.8 |                         0.038 |         1876.06  |          6950.88 |              52.419 |
| A/B: RAG + prompt rag_v2       |                         0.942 |                     0.817 |                               0   |                         0     |         2927.55  |         10801    |              60.452 |
