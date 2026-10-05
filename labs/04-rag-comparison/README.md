# Lab 04 — Vector RAG, GraphRAG, and WikiRAG

This lab implements three retrieval-augmented generation retrieval paradigms on the same fictional university corpus and evaluates them with document-level evidence labels.

- **Vector RAG:** sentence-transformer chunk retrieval, with an optional character-bigram/vector reciprocal-rank fusion (RRF) variant.
- **GraphRAG:** entity-relation triples, local graph expansion for factual and multi-hop questions, and community-summary routing for global questions.
- **WikiRAG:** topic entries that aggregate evidence from several source documents before retrieval.

The script contains 15 base documents, 20 labeled questions (7 factual, 7 multi-hop, and 6 global), the graph triples, and the wiki entries required to reproduce the comparison. The corpus and all names are fictional.

## Setup

```bash
python -m pip install -r requirements.txt
```

The reported experiment used `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. If it cannot be loaded, the script prints a warning and falls back to deterministic character 3-gram vectors; this fallback is useful for smoke tests but does not reproduce the table below.

## Reproduce

```bash
# Main 20-question evaluation at K=5
python hw4_rag.py --mode eval --k 5 --save-json results/eval_k5_20q.json

# K = 1, 2, 3, 5 sensitivity scan on the original 15 questions
python hw4_rag.py --mode eval --scan-k --save-json results/k_scan_builtin15.json

# Optional Vector RAG bigram + vector RRF comparison
python hw4_rag.py --mode eval --k 5 --hybrid --save-json results/eval_k5_20q_hybrid.json

# Inspect one retrieval path or entry set
python hw4_rag.py --mode graph --q Q01 --scope local --k 5
python hw4_rag.py --mode wiki --q Q11 --scope global --k 5

# Fast deterministic smoke test without a model download
python hw4_rag.py --mode eval --k 5 --force-fallback
```

Generation and automatic triple extraction are optional. Set `LLM_API_BASE`, `LLM_API_KEY`, and `LLM_MODEL` only in the environment; never put credentials in this repository.

## Results

Document-level results from the real multilingual sentence embedding run (`K=5`, 20 questions):

| System | Fact Recall@5 / MRR | Multi-hop Recall@5 / MRR | Global Recall@5 / MRR |
| --- | ---: | ---: | ---: |
| Vector RAG | 1.000 / 1.000 | 0.952 / 0.929 | 0.861 / 1.000 |
| GraphRAG | 0.857 / 0.857 | 1.000 / 0.929 | 0.917 / 0.589 |
| WikiRAG | 0.857 / 0.595 | 0.905 / 0.905 | 0.611 / 0.347 |

The results illustrate a trade-off rather than a universal winner: Vector RAG ranks direct factual evidence very well, GraphRAG has the best evidence coverage for multi-hop and global questions, and WikiRAG is compact but sensitive to entry granularity and the Top-K entry window. The optional RRF run did not improve this small corpus (multi-hop Recall@5 fell from 0.952 to 0.905 and global Recall@5 from 0.861 to 0.806).

## Notes

- Metrics are computed after mapping chunks, graph paths, or wiki entries back to source document IDs.
- Graph scope is selected by question type during evaluation: local for factual/multi-hop questions and global for global questions.
- Retrieval quality and answer faithfulness are distinct. This public artifact focuses on reproducible retrieval; the optional API call generates a short evidence-cited answer when credentials are supplied.
