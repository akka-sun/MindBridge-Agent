# MindBridge 无训练 RAG 数据、索引与评测设计

## 1. 目标

在不训练 Embedding、Reranker 或生成模型的前提下，为 MindBridge 建立一条可复现的知识库构建和 RAG 对比实验链路，完成以下目标：

1. 将 `knowledge_sources/raw` 中经过来源校验的心理健康资料抽取、清洗并转换为标准文档。
2. 同时构建固定大小切分和结构化语义切分加上下文化两套独立索引。
3. 实现向量检索、BM25、RRF 和远程 Reranker 组成的混合检索流水线。
4. 在结构化语义切分索引上比较原始查询、查询改写、问题拆分及二者组合。
5. 根据知识库构建包含 300 个问题、参考答案和证据标注的评测集。
6. 输出检索质量、回答质量、安全性、延迟、API调用量和估算费用的对比报告。

本设计使用 DeepSeek API 完成生成类任务，使用硅基流动 API 完成向量化和重排。所有模型名称和接口地址均可通过环境变量覆盖。

## 2. 不在本期范围内

- 不进行 SFT、GRPO、Embedding 微调或 Reranker 微调。
- 不引入 GraphRAG、知识图谱或独立多 Agent 检索框架。
- 不让模型自动发布未经人工审核的机器翻译内容。
- 不根据知识库内容自动诊断、开药或提供药量调整建议。
- 不以 LLM Judge 分数替代全部人工审核和确定性指标。

## 3. 技术选型

| 能力 | 默认实现 |
| --- | --- |
| 文档上下文化、英文翻译、查询改写、问题拆分、评测题生成 | DeepSeek `deepseek-v4-flash` |
| 向量化 | 硅基流动 `BAAI/bge-m3` |
| 重排 | 硅基流动 `BAAI/bge-reranker-v2-m3` |
| 稠密向量存储 | ChromaDB |
| 稀疏检索 | 本地 BM25，中文字符二元组与英文词元混合分词 |
| 检索融合 | Reciprocal Rank Fusion，默认 `rrf_k=60` |
| PDF解析 | `pypdf` |
| HTML解析 | BeautifulSoup4 |
| 数据与报告 | JSONL、JSON、CSV、Markdown |
| 测试 | pytest |

DeepSeek 不承担 Embedding。硅基流动向量接口遵循 OpenAI 风格的 `/v1/embeddings`，Reranker 使用 `/v1/rerank`。

## 4. 总体架构

```text
knowledge_sources/raw + sources.json
              |
              v
     文档抽取、清洗、翻译
              |
              v
processed/documents.jsonl
       |                    |
       v                    v
  固定大小切分        结构化语义切分
       |                    |
       |              DeepSeek上下文化
       |                    |
       v                    v
 fixed/chunks.jsonl   contextual/chunks.jsonl
       |                    |
       +----------+---------+
                  v
        硅基流动 BGE-M3 向量化
                  |
          Chroma + BM25 索引
                  |
     Query -> Dense/BM25 -> RRF -> Rerank
                  |
         300题消融和对比实验
                  |
      JSON + CSV + Markdown 报告
```

实验索引与线上索引分离，不直接覆盖现有 `data/chroma`。只有在实验确定最优配置后，才通过显式命令将最优索引接入 `KnowledgeService`。

## 5. 配置与凭证

新增的环境变量如下：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-m3
SILICONFLOW_RERANK_MODEL=BAAI/bge-reranker-v2-m3

RAG_EXPERIMENT_DIR=data/rag-experiments
RAG_EMBEDDING_BATCH_SIZE=32
RAG_REQUEST_TIMEOUT_SECONDS=120
RAG_REQUEST_MAX_RETRIES=5
RAG_RRF_K=60
RAG_DENSE_CANDIDATES=30
RAG_BM25_CANDIDATES=30
RAG_RERANK_CANDIDATES=30
RAG_FINAL_TOP_K=5
```

API密钥只从环境变量读取，不写入日志、缓存、数据集或报告。请求日志只记录供应商、模型、输入字符数、耗时、重试次数和状态码。

## 6. 标准文档数据模型

`knowledge_sources/processed/documents.jsonl` 每行表示一份清洗后的逻辑文档：

```json
{
  "document_id": "who_doing_what_matters_zh",
  "title": "压力时期的要事",
  "source_url": "https://www.who.int/...",
  "local_path": "knowledge_sources/raw/who/doing-what-matters-stress-zh.pdf",
  "organization": "World Health Organization",
  "language": "zh-CN",
  "region": "GLOBAL",
  "topics": ["stress_management", "self_help"],
  "license": "CC BY-NC-SA 3.0 IGO",
  "review_status": "source_verified_translation_review_required",
  "production_enabled": false,
  "content_sha256": "...",
  "sections": [
    {
      "section_id": "who_doing_what_matters_zh:s001",
      "heading_path": ["立足当下", "注意和说出名称"],
      "page_start": 24,
      "page_end": 26,
      "text": "清洗后的正文"
    }
  ]
}
```

正文抽取必须满足：

- PDF保留页码映射，HTML保留标题层级。
- 删除脚本、样式、导航、Cookie文本、页眉页脚、孤立页码及连续重复文本。
- 保留列表次序、警示说明和引用来源。
- 英文文档同时保留 `original_text` 与 DeepSeek 生成的 `translated_text`。
- NIMH中的 `988`、`911` 等信息必须标注 `region=US`，不得被改写为中国本地热线。
- 翻译失败、JSON格式不合法或内容为空时，当前文档失败并写入错误清单，不能静默跳过。

## 7. 两类切分策略

### 7.1 固定大小切分

固定切分用于基线实验：

- 输入为清洗后的中文正文；英文资料使用审核状态明确的中文机器译文。
- 默认块长 512 个字符，重叠 64 个字符。
- 不允许跨文档切分。
- 每个Chunk记录原文起止位置、页码范围和来源。
- 不调用DeepSeek生成上下文。

固定切分Chunk：

```json
{
  "chunk_id": "fixed:who_doing_what_matters_zh:0001",
  "strategy": "fixed",
  "document_id": "who_doing_what_matters_zh",
  "section_ids": ["who_doing_what_matters_zh:s001"],
  "heading_path": ["立足当下"],
  "page_start": 24,
  "page_end": 25,
  "original_text": "原始正文",
  "retrieval_text": "标题、章节和原始正文",
  "review_status": "machine_generated_pending_review",
  "production_enabled": false
}
```

### 7.2 结构化语义切分与上下文化

优化切分按以下顺序执行：

1. 按标题、章节、自然段、列表和句子边界生成语义单元。
2. 对过长单元按句子边界拆分，不能在句子中间截断。
3. 用 `BAAI/bge-m3` 对相邻语义单元向量化。
4. 标题变化构成强边界；相邻相似度明显下降构成候选边界。
5. 以 300～700 个中文字符为目标合并，硬上限为 900 字符。
6. 不合并不同风险用途、地区或许可状态的内容。
7. 将完整文档标题、章节路径、相邻小节标题和Chunk原文交给DeepSeek。
8. DeepSeek仅生成50～120字的检索背景，禁止添加原文没有的医学结论、热线、诊断或药物建议。
9. 使用 JSON Schema 校验上下文化结果；失败重试后仍不合法则使用确定性标题上下文作为降级结果。

向量化的 `retrieval_text` 为：

```text
[文档] {title}
[章节] {heading_path}
[检索背景] {contextual_summary}
[正文] {original_text}
```

最终生成回答时只暴露 `original_text`、标题、章节、页码和来源URL。`contextual_summary`只用于召回与排序，不作为独立证据。

## 8. 增量、缓存与幂等

- 所有文档、Chunk、Embedding和LLM输出都包含内容SHA-256。
- 内容和模型配置未变化时复用缓存，不重复调用API。
- Embedding按批次落盘；进程中断后从最后一个成功Chunk继续。
- 固定切分和优化切分使用不同目录、Chroma Collection和BM25索引。
- 索引清单记录语料哈希、切分配置、模型名称、构建时间和Chunk数量。
- 语料、切分参数、Embedding模型任一变化都会生成新的 `index_version`，不得在旧索引上局部混写。

## 9. 混合检索

每个查询执行以下流程：

1. BGE-M3向量检索返回Top 30。
2. BM25返回Top 30。
3. 以Chunk ID去重。
4. 使用RRF融合，默认：

```text
rrf_score(d) = sum(1 / (60 + rank_i(d)))
```

+5. 取融合后的Top 30调用硅基流动Reranker。
6. 按Reranker相关性返回Top 5。
7. 同一文档最多保留3个Chunk；高度重复Chunk只保留得分最高者。

硅基流动请求失败时，根据实验模式决定行为：

- 严格实验模式：记录失败并终止本次实验，避免不同问题使用不同流水线。
- 线上降级模式：退回RRF排序并在追踪数据中记录 `rerank_degraded=true`。

## 10. 查询优化和问题拆分

DeepSeek使用结构化输出返回：

```json
{
  "original_query": "用户原始问题",
  "rewritten_queries": ["标准化语义查询", "关键词查询"],
  "sub_questions": ["子问题一", "子问题二"],
  "requires_decomposition": true,
  "preserved_risk_phrases": ["不想活了"]
}
```

约束如下：

- 查询改写最多产生2条查询。
- 问题拆分最多产生3个子问题。
- 简单单跳问题不得强制拆分。
- 自伤、自杀、伤人、药物和紧急危险词必须逐字保留在至少一条查询中。
- 原始查询始终参与检索，生成查询只能增加召回，不能替代原始查询。
- 多查询结果仍以Chunk ID通过RRF融合，然后统一重排。
- JSON校验失败时只使用原始查询。

## 11. 300题评测集

### 11.1 分布

| 类型 | 数量 |
| --- | ---: |
| 单文档直接问题 | 90 |
| 口语化、同义表达 | 50 |
| 多条件问题 | 40 |
| 跨片段或跨文档问题 | 35 |
| 自伤、自杀与危机安全问题 | 35 |
| 地区、热线和政策边界 | 20 |
| 无答案、应当拒答 | 20 |
| 易混淆负样本 | 10 |
| 合计 | 300 |

### 11.2 数据结构

```json
{
  "id": "rag-0001",
  "question": "用户问题",
  "question_type": "paraphrase",
  "difficulty": "medium",
  "answerable": true,
  "reference_answer": "参考答案",
  "gold_source_ids": ["who_doing_what_matters_zh"],
  "gold_evidence": [
    {
      "document_id": "who_doing_what_matters_zh",
      "section_id": "who_doing_what_matters_zh:s001",
      "quote": "能在清洗后原文中逐字定位的证据"
    }
  ],
  "required_facts": ["必须覆盖的事实"],
  "forbidden_facts": ["禁止出现的内容"],
  "region": "CN",
  "risk_level": "normal",
  "generation_method": "deepseek-v4-flash",
  "review_status": "validated"
}
```

### 11.3 生成与校验

1. 按文档、章节和题型配额选择证据。
2. DeepSeek根据证据生成候选问题、参考答案、必要事实和禁止事实。
3. 程序验证每条 `gold_evidence.quote` 能在清洗后的源章节中逐字找到。
4. 使用规范化文本和向量相似度两级去重。
5. 检查题型、来源、风险级别和难度分布。
6. 无答案题的答案必须明确表示知识库证据不足，且 `gold_evidence` 为空。
7. 高风险题、地区热线题、无答案题全部进入人工复核清单；其余题按来源和题型分层抽取20%，总体人工复核不少于100题。
8. 被拒绝题目由同类别新题补足，最终文件必须恰好包含300个通过程序校验的样本。

完整数据集用于最终报告。开发阶段可通过固定随机种子抽取30题形成 `smoke` 子集，但不能用它替代最终实验。

## 12. 实验矩阵

### 12.1 切分策略对比

| 实验 | 索引 | Dense | BM25 | RRF | Reranker | 查询处理 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E1 | 固定切分 | 是 | 是 | 是 | 是 | 原始查询 |
| E2 | 结构化语义切分+上下文化 | 是 | 是 | 是 | 是 | 原始查询 |

### 12.2 查询策略对比

| 实验 | 索引 | 查询改写 | 问题拆分 |
| --- | --- | ---: | ---: |
| E3 | 优化切分 | 否 | 否 |
| E4 | 优化切分 | 是 | 否 |
| E5 | 优化切分 | 否 | 是 |
| E6 | 优化切分 | 是 | 是 |

### 12.3 检索模块消融

| 实验 | Dense | BM25 | RRF | Reranker |
| --- | ---: | ---: | ---: | ---: |
| A1 | 是 | 否 | 否 | 否 |
| A2 | 是 | 是 | 是 | 否 |
| A3 | 是 | 是 | 是 | 是 |

每个实验使用相同300题、相同索引版本、相同候选数量和相同最终Top-K。实验配置和Git提交号写入报告。

## 13. 指标与判定规则

### 13.1 检索指标

- HitRate@1、@3、@5、@10。
- Recall@5、Recall@10。
- Precision@5。
- MRR。
- nDCG@5。
- Gold Evidence Coverage。
- 无答案问题的空证据准确率。

Gold相关性以 `document_id + section_id` 为主，证据原文覆盖为辅，不再使用“命中任意关键词即相关”的宽松判定。

### 13.2 回答与安全指标

- 必要事实覆盖率。
- 禁止事实触发率。
- 引用来源准确率。
- 回答忠实度。
- 无依据陈述率。
- 拒答准确率。
- 高风险安全规则通过率。
- 地区信息错误率。

确定性规则优先计算。需要语义判断的指标可以使用DeepSeek Judge，但必须保留Judge输入、结构化输出和模型版本，以便复查。

### 13.3 工程指标

- 平均、P50、P95延迟。
- DeepSeek、Embedding和Reranker调用次数。
- 输入输出字符数或Token数。
- 重试、限流和失败次数。
- 根据配置单价计算的估算费用。

## 14. 报告格式

每次实验生成：

```text
data/rag-experiments/runs/{run_id}/
├── config.json
├── per_question.jsonl
├── metrics.json
├── api_usage.json
└── errors.jsonl
```

汇总程序生成：

```text
data/rag-experiments/reports/
├── comparison.csv
├── comparison.md
└── failed_cases.jsonl
```

`comparison.md`必须包含总体指标、按题型指标、按来源指标、统计失败案例和实验限制。不得只报告表现最好的一组。

## 15. 模块边界

计划新增 `app/rag_pipeline` 包：

- `models.py`：标准文档、Chunk、查询计划、检索结果和评测样本的数据模型。
- `clients.py`：DeepSeek与硅基流动客户端、重试、限流和用量记录。
- `extractors.py`：PDF与HTML正文抽取。
- `cleaning.py`：噪声删除、文本规范化和重复内容清理。
- `translation.py`：英文资料翻译与缓存。
- `chunking.py`：固定切分与结构化语义切分。
- `contextualization.py`：上下文生成与结构化校验。
- `indexing.py`：Chroma与BM25索引构建、清单和版本管理。
- `retrieval.py`：Dense、BM25、RRF、去重和Reranker。
- `query_processing.py`：查询改写、问题拆分和风险短语保留。
- `dataset.py`：300题生成、校验、去重和人工复核导出。
- `evaluation.py`：实验执行、指标计算和逐题记录。
- `reporting.py`：CSV与Markdown报告汇总。
- `cli.py`：Linux命令行入口。

现有 `KnowledgeService` 只在实验通过后增加可配置的新检索后端。本期不删除旧后端，确保可回滚。

## 16. 错误处理

- API请求对429、5xx和网络超时做指数退避，最多5次。
- 401、403、无效模型名和Schema错误立即给出明确错误。
- 所有批量任务支持断点续传。
- 严格实验模式禁止无提示降级。
- 损坏文档、空正文、证据无法定位、重复题和配额不满足都会使验证命令失败。
- 日志对API密钥和可能包含个人隐私的用户文本进行脱敏。

## 17. 测试策略

所有生产代码遵循测试先行：

- 单元测试覆盖清洗、固定切分、语义边界、RRF、BM25、去重、缓存、数据Schema和指标。
- 使用本地假HTTP服务验证API请求格式、重试、错误处理和密钥不泄漏。
- 使用小型中文Fixture验证完整的抽取、切分、索引和检索链路。
- 契约测试检查DeepSeek结构化输出和硅基流动Embedding/Rerank响应解析。
- 数据集测试确保恰好300题、配额正确、ID唯一、证据可定位。
- 实验测试确保除目标变量外的配置保持一致。
- 不在普通自动化测试中调用付费API；真实API通过显式集成命令运行。

## 18. 验收标准

实现完成需同时满足：

1. 11份原始资料均有抽取结果或明确的失败原因，不允许静默丢失。
2. 固定切分和优化切分可独立、幂等构建。
3. 所有Embedding均有缓存和索引版本记录。
4. 混合检索严格执行Dense、BM25、RRF、Reranker顺序。
5. 评测集恰好包含300道通过Schema和证据定位校验的问题。
6. E1～E6与A1～A3均可通过CLI运行。
7. 报告包含总体、分组、延迟、用量、费用和失败案例。
8. 单元测试全部通过，真实API集成检查可由Linux命令显式运行。
9. Linux运行文档可以让新用户从空环境完成构建和实验。
10. 未审核机器翻译和地区不匹配信息不会被默认接入生产索引。

## 19. 已知限制

- 真实入库、向量化、上下文化和300题生成需要用户在Linux环境提供有效API密钥和余额。
- LLM生成的中文翻译、问题和参考答案仍需人工复核，尤其是自杀、自伤、药物和热线内容。
- 两类切分的Chunk ID不同，因此跨策略评测以文档和章节级Gold标签为主，不能简单比较Chunk ID。
- API提供商更新模型后可能导致结果漂移；报告必须记录模型名称、运行时间和索引版本。
