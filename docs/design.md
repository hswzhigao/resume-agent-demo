# resume-agent-demo 设计说明

> 输入一份 JD，从简历库中语义检索最匹配的候选人并给出带引用的解释。
> 兼作求职作品集：演示 RAG、混合召回、LangGraph 编排、MCP 工具与工程化测试。

## 目标与非目标

**目标**

1. 端到端可跑：解析 → 切分 → 向量化 → 混合召回 → 重排（可选）→ 引用式回答
2. 无 API Key 也能完整跑通（确定性降级路径），便于 CI 与离线演示
3. LangGraph 编排「建查询 → 检索 → 评分 → 解释」四步
4. 以 MCP 工具形式暴露给 Claude Code 等 Agent 调用
5. 输出默认对手机号/邮箱打码（数据安全意识）

**非目标**：多用户鉴权、生产级 ANN 索引调优、模型微调、海量简历性能。

## 架构

```
JD / Query
   │
   ▼
[LangGraph 图]
 build_query ──► retrieve ──► score ──► explain
 (LLM/启发式)     (混合召回)   (LLM/启发式)  (引用)
   │
   ▼
[FastAPI]  /ingest  /search  /match  /health
   │                     │
   ▼                     ▼
[PostgreSQL + pgvector] ◄── ingest: 解析(md/txt/pdf) → 切分 → embedding
   (向量余弦 + pg_trgm 关键词，RRF 融合)

[MCP stdio server] search_resumes / explain_match
```

## 关键决策

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 向量库 | PostgreSQL + pgvector | 面试岗位普遍要求 PG，一鱼两吃 |
| 关键词召回 | pg_trgm 相似度 + token 覆盖率 | 中文无分词器，trigram 对 CJK 子串有效 |
| 融合 | RRF（k=60） | 无需调权重的排名融合，可单测 |
| Embedding | OpenAI 兼容 /embeddings；无 key 时本地 256 维 hash 向量 | 离线可跑、CI 无密钥；切换模型需 `--reset` 重建 |
| LLM 节点 | 无 key 时启发式降级（token 覆盖率评分 + 模板解释） | 图结构始终可验证 |
| PII | 输出侧正则打码，可关 | 默认安全，演示数据安全意识 |
| 测试 | 纯函数单测 + Testcontainers 集成测试；CI 用 pgvector service 容器 | 真实数据库行为进测试 |

## 数据流

1. `ingest data/sample`：解析 → 按标题/段落切分（默认 600 字符、80 重叠）→ embedding → 写 documents/chunks（含向量与 trgm 索引）
2. `search(q)`：q 向量化 → 向量 top30 ∪ 关键词 top30 → RRF 融合 → top_k → 打码输出
3. `match(jd)`：LangGraph 四节点；评分 0-100 + 引用编号；无 LLM 时降级并标注 `mode=heuristic`

## API

- `POST /ingest` `{folder?, reset?}` → `{docs, chunks}`
- `POST /search` `{query, top_k?}` → 候选 chunk + 引用
- `POST /match` `{jd, top_k?}` → 排序候选人 + 解释
- `GET /health` → `{ok, embedding, llm, mode}`

## 已知取舍

- 向量列维度在建表时固定，切换 embedding 模型需 `--reset`（元信息表存 dim 做校验）
- 小数据量用精确扫描，未建 IVFFlat/HNSW（数据到 10 万级再考虑）
- rerank 为可选增强，接口按 Jina/Cohere 风格 `/rerank`，失败静默跳过
