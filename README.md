# MindBridge Agent

面向校园心理健康咨询的多 Agent 平台，围绕心理知识检索、用户意图与情绪识别、会话记忆和风险处理构建咨询链路，提供 SSE 流式对话、学生与管理员入口，以及报告生成和风险预警。

本版本将联合分类 SFT 工程、心理健康知识源和 RAG 对比实验纳入同一仓库，支持从数据准备到训练、检索评测的持续迭代。

## 核心能力与升级

- **心理知识 RAG**：扩充 WHO、NIMH 等机构的心理健康资料，记录来源、校验信息与使用限制；支持固定大小切分和带背景信息的语义切分。
- **混合检索与重排**：实验模块支持向量检索、BM25、RRF 融合和 SiliconFlow Reranker，提供 DeepSeek 查询改写与问题拆分入口。
- **意图、情绪与风险联合识别**：统一输出 `intent / emotion / risk`，供路由与回复策略使用；高风险规则优先，模型输出异常时回退到规则判断。
- **联合分类 SFT**：提供普通与边界问题生成、联合标注、质量过滤、人工抽查、数据划分、QLoRA 训练和训练前后评测脚本。
- **记忆与上下文管理**：MySQL 保存完整记录，Redis 保存近期对话，通过窗口裁剪与记忆摘要控制上下文长度，记忆读写包含隐私脱敏处理。
- **多 Agent 协作**：Memory、Supervisor、Knowledge、RiskGuardian、Companion、Counselor 分工完成记忆加载、意图路由、检索、风险评估和回复规划，支持 LangGraph 与自研 Runtime。
- **风险处理闭环**：报告、Excel 台账和邮件预警通过 MCP 工具与异步队列处理，支持重试、限流及死信记录。

## 技术栈

```text
服务与界面：Python / FastAPI / Uvicorn / SSE / HTML · CSS · JavaScript
Agent 编排：LangGraph / 自研 Runtime
存储与记忆：MySQL / SQLAlchemy / Redis
在线知识库：Chroma / Embeddings / 本地检索兜底
RAG 实验：BM25 / 向量索引 / RRF / SiliconFlow Embedding 与 Reranker
模型接入：Ollama / OpenAI-compatible API / DeepSeek
训练与评测：联合分类 SFT / QLoRA / pytest
工具与部署：MCP / Docker Compose / SMTP / openpyxl
```

在线聊天当前使用 `app/services/knowledge.py`；升级检索与对比实验位于 `app/rag_pipeline/`，通过独立 CLI 运行。实验索引不会自动替换在线 Chroma 知识库。仓库提供联合分类与 SFT 工程，不附带训练后的权重或已验证的 SFT 提升指标。

## 对话流程

```text
用户输入 → 隐私脱敏 → 记忆加载 → 联合分类与意图路由
                                      ↓
                            按需检索心理知识与风险评估
                                      ↓
                       Companion / Counselor → SSE 回复
                                      ↓
                          报告 → MCP 工具队列 → 风险预警
```

## 目录结构

```text
app/
├── agents/          # Agent 编排与 Runtime
├── api/             # 聊天、管理和知识库接口
├── core/            # 配置、数据库与认证
├── services/        # 联合分类、记忆、知识库、报告与工具队列
├── rag_pipeline/    # 语料处理、切分、索引、检索与实验运行器
├── rag_eval/        # 在线知识服务的基础检索评测
├── mcp_tools/       # Excel 与预警工具
└── static/          # 学生与管理员界面
SFT/                 # 联合分类数据、训练与评测工程
knowledge_sources/   # 知识源清单、原始资料与使用说明
docs/                # RAG 运行指导、设计与实施文档
scripts/             # 启动、模型接入及实验脚本
```

## 快速启动（Linux）

```bash
git clone https://github.com/akka-sun/MindBridge-Agent.git
cd MindBridge-Agent
cp .env.example .env
```

在 `.env` 中设置模型连接、数据库和 Redis 配置。使用 Docker Compose 启动：

```bash
docker compose up -d --build
```

应用端口为 `8080`，MySQL 和 Redis 的宿主机端口分别为 `13306`、`16379`。默认聊天模型通过宿主机 Ollama 提供，需准备模型或配置其他 Provider；仅验证流程可使用 `AI_PROVIDER=mock`。

Ollama 默认模型为 `qwen2.5:7b`，先在宿主机执行 `ollama pull qwen2.5:7b`；使用其他模型时设置 `OLLAMA_MODEL`。SFT 训练输出由 `SFT/` 管理，不依赖旧版 GGUF 目录。

本地启动：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 配置并启动 MySQL、Redis 后执行
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

默认演示账号：学生 `student / student123`，管理员 `admin / admin123`。部署前修改默认凭据。

```bash
curl -N -u student:student123 \
  -H 'Content-Type: application/json' \
  -d '{"message":"最近考试压力很大，晚上总是睡不着"}' \
  http://127.0.0.1:8080/api/chat/stream
```

## RAG 构建与对比实验

配置 `DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY` 及对应模型名称后执行：

```bash
python -m app.rag_pipeline.cli validate-sources
python -m app.rag_pipeline.cli build-corpus
python -m app.rag_pipeline.cli build-chunks --strategy fixed \
  --output data/rag-experiments/chunks/fixed.jsonl
python -m app.rag_pipeline.cli build-chunks --strategy contextual \
  --output data/rag-experiments/chunks/contextual.jsonl
```

实验覆盖切分策略、重排、查询改写与问题拆分，输出每题检索记录及 chunk 级 HitRate、MRR、Recall、Precision。运行步骤见 [RAG 指导](docs/RAG_IMPLEMENTATION.md)，知识来源见 [知识源说明](knowledge_sources/README.md)。

历史实验包含离线配置、不同问题集及查询规划降级情况，暂不作为严格的同条件提升结论。比较切分策略时，应使用相同问题和证据，并分别映射到各索引的 gold chunk；仅有配置或指标文件不能证明查询改写、拆分实际生效。

已移除旧版 `make-300-dataset` 模板出题入口。评测问题需逐题检查语义、答案与证据支持，不能通过模板长度或统一标记 `hard` 认定难度。已有本地数据和实验记录保留，便于复核。

## 联合分类 SFT

训练与评测入口位于 `SFT/`，支持问题生成、联合标注、人工审核、数据划分、QLoRA 训练与模型对比。Linux 环境配置和命令见 [SFT 运行指南](SFT/README.md)，实施方案见 [SFT 实施文档](SFT/IMPLEMENTATION_PLAN.md)。

## 工程验证

```bash
python -m pytest -q
AI_PROVIDER=mock python -m app.rag_eval.runner
```

API 密钥、训练数据、权重和本地实验产物不提交到仓库。心理支持内容用于辅助咨询，高风险情况通过安全规则和人工支持流程处理。
