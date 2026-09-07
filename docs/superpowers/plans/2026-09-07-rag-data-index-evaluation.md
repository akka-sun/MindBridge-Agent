# MindBridge RAG Data, Indexing, and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Linux-runnable, resumable RAG corpus pipeline with two chunking strategies, SiliconFlow embeddings/reranking, Dense+BM25+RRF retrieval, DeepSeek query planning, a validated 300-question dataset workflow, and reproducible comparison reports.

**Architecture:** Add an isolated `app.rag_pipeline` package and CLI without replacing the current production `KnowledgeService`. Raw WHO/NIMH/government files become normalized JSONL documents, then two versioned chunk/index variants. Experiments consume immutable index manifests and dataset manifests so every comparison changes only the intended variable.

**Tech Stack:** Python 3.10+, Pydantic, httpx, pypdf, BeautifulSoup4, ChromaDB, rank-bm25, numpy, pytest, DeepSeek V4 API, SiliconFlow Embedding and Rerank APIs.

**Spec:** `docs/superpowers/specs/2026-09-07-rag-data-index-evaluation-design.md`

## Global Constraints

- Linux is the only documented execution environment.
- DeepSeek default model is `deepseek-v4-flash`.
- SiliconFlow defaults are `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`.
- API keys are read only from environment variables and never persisted.
- Unit tests never call paid APIs.
- Fixed and contextual indices are physically and logically isolated.
- Machine-generated translations and safety-sensitive data remain non-production until reviewed.
- All API work is cached by content, prompt, schema, model, and parameter hashes.

---

### Task 1: Package Configuration and Core Models

**Files:**
- Create: `app/rag_pipeline/__init__.py`
- Create: `app/rag_pipeline/config.py`
- Create: `app/rag_pipeline/models.py`
- Create: `tests/rag_pipeline/test_models.py`
- Modify: `requirements.txt`
- Modify: `.env.example`

**Interfaces:**
- Produces: `RagPipelineSettings`, `SourceDocument`, `DocumentSection`, `KnowledgeChunkRecord`, `QueryPlan`, `RetrievedChunk`, `EvaluationCase`, `ExperimentConfig`.

- [ ] Write tests that reject invalid page ranges, empty evidence quotes, overlong query plans, and experiment configs with inconsistent retrieval flags.
- [ ] Run `pytest tests/rag_pipeline/test_models.py -q` and verify missing-module failure.
- [ ] Implement Pydantic models and environment-backed settings with resolved project paths.
- [ ] Add exact dependencies and documented environment names.
- [ ] Run the tests and commit.

### Task 2: Versioned Prompts and JSON Schemas

**Files:**
- Create: `app/rag_pipeline/prompts/*.md`
- Create: `app/rag_pipeline/schemas/*.schema.json`
- Create: `app/rag_pipeline/prompting.py`
- Create: `tests/rag_pipeline/test_prompting.py`

**Interfaces:**
- Produces: `PromptAsset.load(prompt_id)`, `PromptAsset.cache_fingerprint(model, parameters)`, and `PromptAsset.render(values)`.

- [ ] Write failing tests for missing metadata, wrong schema references, absent ordinary/boundary/failure few-shot examples, unsafe document interpolation, and stable fingerprints.
- [ ] Run tests and verify expected failures.
- [ ] Write seven complete prompts and six strict schemas from the approved spec.
- [ ] Implement loader, schema validation, XML data delimiters, rendering, and fingerprinting.
- [ ] Run tests and commit.

### Task 3: HTTP Clients, Retry, Cache, and Usage Ledger

**Files:**
- Create: `app/rag_pipeline/cache.py`
- Create: `app/rag_pipeline/clients.py`
- Create: `tests/rag_pipeline/test_clients.py`

**Interfaces:**
- Produces: `DeepSeekClient.generate_json(...)`, `SiliconFlowClient.embed(...)`, `SiliconFlowClient.rerank(...)`, `JsonFileCache`, `UsageLedger`.

- [ ] Write tests using `httpx.MockTransport` for exact endpoint payloads, successful parsing, 429/5xx retry, non-retryable 401, malformed responses, cache hits, and secret redaction.
- [ ] Run tests and verify missing implementations fail.
- [ ] Implement synchronous clients with exponential backoff, injected sleep/transport, deterministic cache keys, and append-only usage events.
- [ ] Run tests and commit.

### Task 4: Extraction, Cleaning, Translation, and Document Build

**Files:**
- Create: `app/rag_pipeline/extractors.py`
- Create: `app/rag_pipeline/cleaning.py`
- Create: `app/rag_pipeline/translation.py`
- Create: `app/rag_pipeline/corpus.py`
- Create: `tests/rag_pipeline/fixtures/sample.html`
- Create: `tests/rag_pipeline/test_corpus.py`

**Interfaces:**
- Produces: `extract_source(path)`, `clean_blocks(blocks)`, `TranslationService.translate_section(...)`, `CorpusBuilder.build(manifest_path, output_path)`.

- [ ] Write tests for HTML navigation removal, PDF page mapping, duplicate line removal, heading preservation, source hash verification, English original/translation pairing, and explicit per-source errors.
- [ ] Run tests and verify failures.
- [ ] Implement deterministic extraction/cleaning and DeepSeek-backed cached translation.
- [ ] Implement resumable JSONL document build with a summary and errors file.
- [ ] Run tests and commit.

### Task 5: Fixed and Contextual Semantic Chunking

**Files:**
- Create: `app/rag_pipeline/chunking.py`
- Create: `app/rag_pipeline/contextualization.py`
- Create: `tests/rag_pipeline/test_chunking.py`

**Interfaces:**
- Produces: `FixedChunker.chunk(document)`, `SemanticChunker.chunk(document, embedder)`, `Contextualizer.contextualize(chunks)`.

- [ ] Write tests for fixed 512/64 behavior, no cross-document chunks, sentence-safe semantic boundaries, heading boundaries, 300-700 target sizes, 900 hard limit, and deterministic fallback context.
- [ ] Run tests and verify failures.
- [ ] Implement chunkers and adjacent-paragraph cosine boundary detection.
- [ ] Implement DeepSeek contextualization with `added_claims=false` enforcement and cached fallback.
- [ ] Run tests and commit.

### Task 6: Versioned Dense and BM25 Indices

**Files:**
- Create: `app/rag_pipeline/indexing.py`
- Create: `tests/rag_pipeline/test_indexing.py`

**Interfaces:**
- Produces: `IndexBuilder.build(chunks_path, strategy) -> IndexManifest`, `DenseIndex.search`, `Bm25Index.search`.

- [ ] Write tests for stable index versions, separate collections, batch resume, stale-manifest rejection, Chinese bigram retrieval, and persisted BM25 state.
- [ ] Run tests and verify failures.
- [ ] Implement Chroma collections, embedding checkpoint files, BM25 token statistics, manifests, and reloadable search adapters.
- [ ] Run tests and commit.

### Task 7: Hybrid Retrieval, Query Planning, and Reranking

**Files:**
- Create: `app/rag_pipeline/query_processing.py`
- Create: `app/rag_pipeline/retrieval.py`
- Create: `tests/rag_pipeline/test_retrieval.py`

**Interfaces:**
- Produces: `QueryPlanner.plan(query, history)`, `reciprocal_rank_fusion(rankings, rrf_k)`, `HybridRetriever.retrieve(query, config)`.

- [ ] Write tests for hand-calculated RRF scores, deduplication, document diversity, raw-query preservation, maximum two rewrites/three subquestions, risk phrase preservation, strict reranker failure, and online degradation.
- [ ] Run tests and verify failures.
- [ ] Implement multi-query Dense+BM25 retrieval, RRF, candidate truncation, SiliconFlow rerank, and final Top-K selection.
- [ ] Run tests and commit.

### Task 8: 300-Question Dataset Generation and Review Workflow

**Files:**
- Create: `app/rag_pipeline/dataset.py`
- Create: `app/rag_pipeline/dataset_quota.py`
- Create: `tests/rag_pipeline/test_dataset.py`

**Interfaces:**
- Produces: `DatasetGenerator.generate(...)`, `DatasetValidator.validate(...)`, `ReviewExporter.export(...)`.

- [ ] Write tests for exact 300-item quotas, unique IDs, literal evidence location, answerable/unanswerable invariants, normalized and semantic duplicate rejection, and the at-least-100 review sample.
- [ ] Run tests and verify failures.
- [ ] Implement two-stage DeepSeek generation/labeling, deterministic quota scheduling, evidence lookup, refill, validation, and stratified review export.
- [ ] Add a deterministic 300-case seed specification so the final dataset build is reproducible when API keys are supplied.
- [ ] Run tests and commit.

### Task 9: Experiment Runner, Metrics, and Reports

**Files:**
- Create: `app/rag_pipeline/evaluation.py`
- Create: `app/rag_pipeline/reporting.py`
- Create: `app/rag_pipeline/experiments.yaml`
- Create: `tests/rag_pipeline/test_evaluation.py`

**Interfaces:**
- Produces: `ExperimentRunner.run(config, dataset)`, `compute_retrieval_metrics`, `ReportBuilder.build(run_dirs)`.

- [ ] Write tests with hand-calculated HitRate, Recall, Precision, MRR, nDCG, evidence coverage, grouped metrics, latency percentiles, and experiment-variable isolation.
- [ ] Run tests and verify failures.
- [ ] Implement A1-A3 and E1-E6 configs, per-question traces, deterministic metrics, optional DeepSeek judge, usage aggregation, CSV/Markdown reports, and failure-case export.
- [ ] Run tests and commit.

### Task 10: CLI and Linux Runbook

**Files:**
- Create: `app/rag_pipeline/cli.py`
- Create: `scripts/rag-pipeline.sh`
- Create: `docs/RAG_IMPLEMENTATION.md`
- Create: `tests/rag_pipeline/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces commands: `validate-sources`, `build-corpus`, `build-chunks`, `build-index`, `generate-dataset`, `validate-dataset`, `run-experiments`, `build-report`, `run-all`.

- [ ] Write subprocess-level CLI tests for help, missing keys, dry-run, validation failures, and smoke workflow.
- [ ] Run tests and verify failures.
- [ ] Implement CLI orchestration with explicit paid-API confirmation flag and resumable stages.
- [ ] Write Linux-only commands, expected outputs, review gates, costs/usage inspection, troubleshooting, and production promotion steps.
- [ ] Run focused and full test suites and commit.

### Task 11: Real Corpus Dry Run and Final Verification

**Files:**
- Generate when credentials exist: `knowledge_sources/processed/*`
- Generate when credentials exist: `data/rag-experiments/*`
- Modify if necessary: `.gitignore`

**Interfaces:**
- Consumes all previous CLI commands.
- Produces validation evidence and, only with valid keys, real vector indices and the finalized 300-case dataset.

- [ ] Run source validation and extraction locally without paid APIs.
- [ ] Run `run-all --dry-run` and verify planned API calls and paths.
- [ ] If both keys are configured, run the paid build explicitly; otherwise stop at the credential boundary and report the exact remaining Linux commands.
- [ ] Run `pytest -q`, `git diff --check`, and inspect `git status --short`.
- [ ] Use the verification and branch-finishing skills before claiming completion.
