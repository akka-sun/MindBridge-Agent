# RAG构建与实验运行指导（Linux）

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

## 配置API

```bash
export DEEPSEEK_API_KEY='你的DeepSeek密钥'
export SILICONFLOW_API_KEY='你的硅基流动密钥'
export DEEPSEEK_MODEL='deepseek-v4-flash'
export SILICONFLOW_EMBEDDING_MODEL='BAAI/bge-m3'
export SILICONFLOW_RERANK_MODEL='BAAI/bge-reranker-v2-m3'
```

DeepSeek用于翻译、上下文化、查询改写、问题拆分和出题；硅基流动用于Embedding和Reranker。密钥不会写入文件。

## 不调用付费API的资料检查和语料构建

```bash
./scripts/rag-pipeline.sh validate-sources
./scripts/rag-pipeline.sh build-corpus
./scripts/rag-pipeline.sh build-chunks --strategy fixed --output data/rag-experiments/chunks/fixed.jsonl
./scripts/rag-pipeline.sh build-chunks --strategy contextual --output data/rag-experiments/chunks/contextual.jsonl
```

首次构建前请检查 `data/rag-experiments/processed/build-errors.json`。英文资料的机器翻译和上下文化需要在启用API的完整构建命令中执行，并在人工审核后才允许生产索引。

## 运行测试

```bash
python -m pytest -q
```

## 安全要求

- NIMH资料中的美国 `988/911` 不能作为中国用户的本地热线返回。
- 高风险、自伤、自杀和药物相关片段必须经过安全规则审核。
- mhGAP资料不能用于自动诊断、处方或药量调整。
- 真实API实验必须单独记录模型、Prompt版本、索引版本和费用。
