# Machine Learning Notes RAG

一个面向英文机器学习 Markdown 笔记的中文问答 RAG 系统。项目将 `data/ML-Notes-in-Markdown-master` 中的结构化 Markdown 笔记构建为可检索知识库，支持用户用中文提问，并基于检索到的英文笔记内容生成中文回答。

## Highlights

- 面向英文机器学习 Markdown 笔记构建中文问答系统，支持中文问题检索英文资料。
- 实现 Markdown 标题感知分块，并保留 `topic`、`chapter`、`title`、`section_path`、`chunk_id` 等元数据。
- 使用多语 HuggingFace embedding、FAISS 向量检索、BM25 关键词检索和 RRF 融合重排。
- 针对中文问题与英文语料的错配，加入中文到英文 retrieval query 改写。
- 对列表类问题使用 topic 元数据直接列出文档，避免中文 query 在英文语料上召回不全。
- 提供 Streamlit Web UI，展示回答、检索 query、source path 和 top-k chunks。
- 提供 FastAPI 后端接口，并用 JSONL 日志记录 `question`、`retrieval_query`、`top_sources`、`latency_ms`、`stage`、`error` 和检索分项耗时。
- 提供不调用 LLM 的离线检索评估脚本；当前 15 个测试 case 上 Top-1 source accuracy 为 40.0%，Top-3 source accuracy 为 73.3%，Top-3 topic accuracy 为 93.3%。

## 项目目标

机器学习资料常以英文 Markdown、博客或课程笔记形式存在。中文使用者在查找概念、算法步骤、公式解释和注意事项时，通常需要跨语言检索，并且希望能追踪答案来源。

本项目的目标是构建一个小型但完整的 RAG 应用原型，重点关注：

- 如何把英文 Markdown 笔记转成可检索知识库。
- 如何支持中文问题检索英文内容。
- 如何把检索来源暴露给用户，降低生成式回答的不可验证性。
- 如何用离线评估脚本量化检索效果，而不是只依赖主观演示。
- 如何通过 API、错误处理和结构化日志提升 RAG 系统的可调用性与可观测性。

## 数据特点

数据目录：

```text
data/ML-Notes-in-Markdown-master
```

数据是英文机器学习笔记，主要特点：

- 以 Markdown 文件存储。
- 目录天然对应机器学习主题，例如 `Regression`、`Classification`、`Clustering`、`Deep Learning`。
- 文档内部包含多级标题、公式图片链接、列表和代码引用。
- 不同主题内容密度不均，有些目录只有很短的 README，有些主题包含多个算法文件。

## 数据来源与许可

本项目使用的机器学习 Markdown 笔记来自 `ML-Notes-in-Markdown`，原始作者为 Purnesh Tripathi。

该数据使用 MIT License：

- 允许使用、复制、修改和再分发。
- 允许用于个人项目、学习项目和公开 GitHub 仓库。
- 需要保留原始版权声明和 License 文本。

本项目保留了原始 License 文件：

```text
data/ML-Notes-in-Markdown-master/LICENSE
```

如果你重新整理或发布这个项目，请继续保留该 License 文件，并在 README 中说明数据来源。

## 系统架构

```text
Markdown files
    |
    v
DataPreparationModule
    - load *.md
    - extract title/topic/chapter/section metadata
    - split by Markdown headers
    - recursively split long chunks
    |
    v
IndexConstructionModule
    - multilingual HuggingFace embeddings
    - FAISS vector index
    |
    v
RetrievalOptimizationModule
    - FAISS semantic search
    - BM25 keyword search
    - RRF rerank
    - metadata filtering by topic
    |
    v
GenerationIntegrationModule
    - query routing
    - Chinese-to-English retrieval query rewrite
    - Chinese answer generation
    |
    v
FastAPI / Streamlit UI / CLI
```

## 核心实现

### 1. Markdown 结构感知分块

项目不是直接按固定长度切分全文，而是先使用 Markdown 标题层级切分：

- `#`
- `##`
- `###`
- `####`

每个 chunk 会保留章节路径，例如：

```text
Linear Regression > Variable Selection
```

如果某个标题块过长，再使用递归字符切分，避免单个 chunk 过大导致召回或生成阶段上下文质量下降。

### 2. 主题元数据

系统从目录名中提取主题：

```text
01-Regression -> Regression
02-Classification -> Classification
03-Clustering -> Clustering
```

每个文档和 chunk 都会带有：

- `title`
- `topic`
- `chapter`
- `relative_path`
- `section_path`
- `parent_id`
- `chunk_id`

这些字段用于过滤、展示来源和回溯父文档。

### 3. 检索流程

系统的检索链路会根据问题类型走不同路径：

```text
user question
  -> query router: list / detail / general
  -> topic detection: 从中文或英文问题中识别 Regression、Classification、Clustering 等主题
  -> retrieval query:
       list 问题：保留原问题，优先用 topic 元数据列出父文档
       detail/general 问题：调用 LLM 把中文问题改写成英文检索 query
  -> candidate retrieval:
       FAISS semantic search: k=5
       BM25 keyword search: k=5
  -> RRF rerank:
       按 1 / (60 + rank) 融合两路排序
  -> optional metadata filter:
       如果识别到 topic，先取 top_k * 3 候选，再按 metadata.topic 过滤
  -> return top_k chunks
  -> answer generation / source display
```

检索模块采用：

- FAISS：语义相似度检索。
- BM25：关键词检索，适合 `OLS`、`p-value`、`K-means` 这类精确术语。
- RRF：融合两路结果，降低单一路径漏召回风险。

当前实现中，`hybrid_search(query, top_k)` 会先分别取 FAISS top-5 和 BM25 top-5，再做 RRF 融合，并返回最终 top-k。`metadata_filtered_search(query, filters, top_k)` 会先扩大候选到 `top_k * 3`，再用 chunk 元数据过滤，例如只保留 `topic=Classification` 的结果。

列表类问题有一条特殊路径。例如“分类有哪些方法”会先识别到 `Classification` topic，然后直接从父文档元数据列出该主题下的文档，而不是完全依赖向量相似度。这是为了避免中文问题在英文语料上召回不全。

详细问答则会先做中文到英文 query rewrite，例如“线性回归怎么做变量选择？”会被改写成 `linear regression variable selection feature selection p-value backward elimination`，再进入混合检索。

检索模块会统计混合检索内部耗时：

- `faiss_ms`：FAISS 向量检索耗时。
- `bm25_ms`：BM25 关键词检索耗时。
- `rrf_ms`：RRF 融合重排耗时。
- `total_retrieval_ms`：一次 hybrid search 的总检索耗时。

这些指标会写入 JSONL 日志中的 `retrieval_metrics` 字段，便于判断检索阶段的瓶颈是否来自向量检索、关键词检索还是融合排序。

### 4. 中文问题适配英文资料

由于数据是英文 Markdown，而用户主要用中文提问，系统会在详细问答时调用 LLM 将中文问题改写成英文检索 query。例如：

```text
线性回归怎么做变量选择？
```

可能被改写为：

```text
linear regression variable selection feature selection p-value backward elimination
```

对于“分类有哪些方法”这类列表问题，系统不会依赖向量检索，而是直接按 `Classification` 主题列出对应文档，避免英文语料上的中文检索召回不全。

## 目录结构

```text
.
├── code
│   ├── api.py
│   ├── config.py
│   ├── evaluate_retrieval.py
│   ├── main.py
│   ├── rag_logger.py
│   ├── requirements.txt
│   ├── streamlit_app.py
│   └── rag_modules
│       ├── data_preparation.py
│       ├── generation_integration.py
│       ├── index_construction.py
│       └── retrieval_optimization.py
├── data
│   └── ML-Notes-in-Markdown-master
├── docs
│   └── backend_api_logging.md
├── .gitignore
└── README.md
```

## 环境配置

克隆项目：

```bash
git clone https://github.com/find1one/ML-notes-Rag.git
cd ML-notes-Rag
```

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\activate
pip install -r code\requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r code/requirements.txt
```

如果 PyPI 下载较慢，可以使用镜像：

```powershell
pip install -r code\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

生成最终回答、CLI 问答和 FastAPI 后端都需要设置 Moonshot API key：

```powershell
$env:MOONSHOT_API_KEY="your_api_key"
```

macOS / Linux：

```bash
export MOONSHOT_API_KEY="your_api_key"
```

不设置 API key 时，仍然可以运行离线检索评估脚本，也可以在 Streamlit UI 中关闭 LLM answer generation 和 LLM query rewrite，仅查看检索结果。`code/main.py` 和 `code/api.py` 会初始化 LLM，因此无 key 时会直接报错。

## CLI 运行

CLI 会完成知识库构建、query routing、query rewrite、检索和最终回答生成，因此需要提前设置 `MOONSHOT_API_KEY`。

Windows PowerShell：

```powershell
python code\main.py
```

macOS / Linux：

```bash
python code/main.py
```

如果 embedding 模型已经下载到本地缓存，但运行时 Hugging Face 联网检查失败，可以开启离线模式：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python code\main.py
```

首次运行会构建 FAISS 索引并保存到：

```text
code/vector_index_ml_notes
```

该索引是可重新生成的产物，默认不建议提交到 GitHub。

## Streamlit Web UI

项目提供了一个 Streamlit 界面，用于展示回答和检索来源。

Windows PowerShell：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
python -m streamlit run code\streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

macOS / Linux：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m streamlit run code/streamlit_app.py
```

打开浏览器访问：

```text
http://127.0.0.1:8501
```

注意：运行 Streamlit 的终端窗口需要保持打开。关闭窗口后，本地 Web 服务会停止，浏览器会显示 connection refused。

UI 支持：

- 输入中文问题。
- 显示识别到的 query type 和 topic。
- 显示检索 query。
- 展示 top-k chunks、section path 和 source path。
- 可选择是否调用 LLM 生成最终回答。
- 在未设置 API key 时，也可以只看检索结果。

## FastAPI 后端与 JSONL 日志

项目提供 FastAPI 后端入口：

```text
code/api.py
```

主要接口：

```text
GET  /health
GET  /ready
POST /chat
POST /chat/debug
```

本地启动方式。服务启动时会初始化 RAG 系统并加载或构建 FAISS 索引，因此需要提前设置 `MOONSHOT_API_KEY`：

```bash
cd code
uvicorn api:app --reload
```

访问：

```text
http://127.0.0.1:8000
```

接口调用示例：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"线性回归是什么？"}'
curl -X POST http://127.0.0.1:8000/chat/debug \
  -H "Content-Type: application/json" \
  -d '{"question":"线性回归怎么做变量选择？"}'
```

`/chat` 只返回最终回答：

```json
{
  "answer": "..."
}
```

`/chat/debug` 会额外返回 `route_type`、`retrieval_query` 和 `sources`，用于观察 RAG 中间链路：

```json
{
  "answer": "...",
  "route_type": "detail",
  "retrieval_query": "linear regression variable selection feature selection p-value backward elimination",
  "sources": [
    {
      "title": "Linear Regression",
      "topic": "Regression",
      "section": "Linear Regression > Variable Selection",
      "path": "01-Regression/01-LinearRegression.md"
    }
  ]
}
```

后端同时接入了 JSONL 请求日志：

```text
logs/rag_queries.jsonl
```

每条日志记录一行 JSON，核心字段包括：

```text
question
retrieval_query
top_sources
latency_ms
stage
error
retrieval_metrics
```

`retrieval_metrics` 会进一步记录：

```text
faiss_ms
bm25_ms
rrf_ms
total_retrieval_ms
```

示例：

```json
{
  "question": "线性回归是什么",
  "retrieval_query": "linear regression definition overview OLS ordinary least squares",
  "top_sources": [
    {
      "title": "Linear Regression",
      "topic": "Regression",
      "section": "Linear Regression",
      "path": "01-Regression/01-LinearRegression.md"
    }
  ],
  "latency_ms": 138806,
  "stage": "response",
  "error": null,
  "retrieval_metrics": {
    "faiss_ms": 201,
    "bm25_ms": 0,
    "rrf_ms": 0,
    "total_retrieval_ms": 202
  }
}
```

这类日志可以判断一次请求是慢在检索、query rewrite 还是 LLM generation。例如如果 `latency_ms` 远大于 `total_retrieval_ms`，通常说明检索不是主要瓶颈。

错误场景也会记录到日志。例如空问题会返回 `400`，并记录：

```json
{
  "question": "",
  "retrieval_query": null,
  "top_sources": [],
  "latency_ms": 1,
  "stage": "validate",
  "error": "400: Question cannot be empty",
  "retrieval_metrics": null
}
```

更详细的后端接口、错误处理和日志设计说明见：

```text
docs/backend_api_logging.md
```

## 示例

### 示例 1：列表类问题

输入：

```text
分类有哪些方法
```

输出示例：

```text
根据当前笔记，相关条目包括：
1. LogisticRegression（Classification）
2. knn（Classification）
3. SupportVectorMachines（Classification）
4. Naive Bayes（Classification）
5. DecisionTree（Classification）
6. RandomForest（Classification）
7. HiddenMarkovModels（Classification）
8. Classification（Classification）
```

### 示例 2：概念解释

输入：

```text
什么是聚类
```

系统会检索 `Clustering` 主题下的相关章节，并用中文解释聚类的基本概念、适用前提和相关算法注意点。

### 示例 3：算法细节

输入：

```text
线性回归怎么做变量选择？
```

系统会命中：

```text
Linear Regression > Variable Selection
```

并基于笔记解释 All-in、Backward Elimination、Forward Selection、Bidirectional Elimination 等方法。

## 离线检索评估

项目提供了一个不调用 LLM 的离线评估脚本，用来检查检索模块是否能命中预期主题和源文件：

Windows PowerShell：

```powershell
$env:PYTHONIOENCODING='utf-8'
python code\evaluate_retrieval.py --top-k 3
```

macOS / Linux：

```bash
python code/evaluate_retrieval.py --top-k 3
```

脚本会加载已有 FAISS 索引，运行内置测试集，并输出：

- Top-1 source accuracy
- Top-k source accuracy
- Top-k topic accuracy
- 每个 case 的 top-k 检索结果、topic、section 和 source path

当前基线结果（`--top-k 3`）：

```text
Cases: 15
Top-1 source accuracy: 6/15 = 40.0%
Top-3 source accuracy: 11/15 = 73.3%
Top-3 topic accuracy: 14/15 = 93.3%
```

评估指标含义：

| 指标 | 含义 | 当前结果 |
| --- | --- | --- |
| Top-1 source accuracy | 第 1 个 chunk 的 `relative_path` 命中预期源文件 | 40.0% |
| Top-3 source accuracy | 前 3 个 chunk 中任意一个命中预期源文件 | 73.3% |
| Top-3 topic accuracy | 前 3 个 chunk 中任意一个命中预期主题 | 93.3% |

这个结果说明：主题召回整体可用，适合作为 RAG 答案的初筛依据；但精确 source 排序仍有优化空间，Top-1 结果不应被当成唯一依据。实际 UI 和 `/chat/debug` 会展示多个 source，方便用户判断召回是否可靠。

## 失败案例分析

本轮离线评估中，Top-3 source 未命中的 case 有 4 个：

| Case | Query | 预期 source | 实际 Top-3 概况 | 可能原因 |
| --- | --- | --- | --- | --- |
| `decision_tree` | `decision tree classification algorithm` | `02-Classification/05-DecisionTree.md` | 召回 Naive Bayes 和 Classification README | query 中的 `classification algorithm` 与分类总览和 Naive Bayes 文档重叠较强，Decision Tree 文档自身有效文本较短，排序竞争力不足 |
| `random_forest` | `random forest classification ensemble decision trees` | `02-Classification/06-RandomForest.md` | 召回 K-means、Classification README、Support Vector Regression | `forest`、`ensemble` 等关键词在短文档中支撑不足，语义检索被其他算法片段干扰，缺少强 metadata 约束 |
| `gaussian_mixture` | `gaussian mixture model clustering expectation maximization` | `03-Clustering/03-GaussianMixtureModels.md` | 召回 Hierarchical Clustering、Linear Regression、Clustering README | 同属 Clustering 的概览内容被排到前面，目标文档可能因 chunk 内容短或标题权重不足没有进入 Top-3 |
| `numpy_matrix` | `numpy matrix tutorial Python` | `00-Prerequisites/numpyMatrixTutorial.md` | 召回 Appendix 下的 Numpy 文档 | 语义上 Appendix Numpy 文档确实相关，但评估预期是 Prerequisites 中的旧教程文件，说明数据集中存在主题重复和路径命名不一致 |

这些失败不代表系统完全无法回答对应问题，而是说明“精确源文件命中”仍不稳定。当前缓解方式包括：

- 对列表类问题优先走 topic 元数据和父文档列表。
- 在 UI 和 `/chat/debug` 中展示多个 source，而不是只展示 Top-1。
- 使用 BM25 + FAISS + RRF 融合，减少单一路径召回偏差。
- 在后续优化中给标题、文件名和 topic metadata 更高权重，尤其针对短文档和算法名明确的问题。

## Screenshots

### Streamlit UI

![Streamlit UI](docs/images/ML-Notes-RAG.png)

### Offline Retrieval Evaluation

![Offline retrieval evaluation](docs/images/evaluate_retrival.png)

## 当前限制

这个项目不是完整的机器学习教材问答系统，当前限制包括：

- 当前评估集规模较小，仅包含 15 个离线检索测试 case。
- 生成质量依赖 Moonshot/Kimi API；未设置 API key 时无法生成最终 LLM 回答，但仍可运行检索、离线评估和 Streamlit 检索预览。
- Markdown 中的图片公式目前只保留链接文本，尚未做 OCR 或公式解析。
- 部分短文档或弱关键词文档，如 Decision Tree、Random Forest、Gaussian Mixture，仍存在精确 source 命中不稳定的问题。
- 当前 Streamlit UI 和 FastAPI 后端主要用于本地演示，尚未做部署、鉴权或多用户并发支持。

## 后续计划

优先级较高的改进：

1. 扩大离线评估集，从 15 个 case 扩展到 50 个以上。
2. 对比不同 top-k、chunk size、RRF 参数对 Top-1 / Top-k accuracy 的影响。
3. 增加 Base LLM vs RAG 的回答质量对比。
4. 优化短文档和弱关键词文档的 metadata filtering。
5. 补充更多典型问答案例截图，覆盖 list、detail、LLM 失败 fallback 等场景。
6. 清理标题展示格式，例如 `LogisticRegression` -> `Logistic Regression`。
7. 将 `/chat` 也接入完整 JSONL 日志，并继续细化 `rewrite_ms`、`generation_ms` 等阶段耗时。
