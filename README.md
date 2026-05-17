# Machine Learning Notes RAG

一个面向英文机器学习 Markdown 笔记的中文问答 RAG 系统。项目将 `data/ML-Notes-in-Markdown-master` 中的结构化 Markdown 笔记构建为可检索知识库，支持用户用中文提问，并基于检索到的英文笔记内容生成中文回答。

## 项目目标

这个项目解决的问题是：机器学习资料通常以英文 Markdown、博客或课程笔记形式存在，中文使用者在查找概念、算法步骤和注意事项时，需要跨语言检索并追踪答案来源。

系统当前支持：

- 递归加载 Markdown 笔记。
- 按 Markdown 标题结构进行语义分块。
- 对过长标题块做二次切分，避免上下文过长。
- 提取 `topic`、`chapter`、`title`、`section_path` 等元数据。
- 使用多语 embedding 支持中文问题检索英文资料。
- 使用 FAISS 向量检索、BM25 关键词检索和 RRF 融合重排。
- 对列表类问题直接按主题列出文档，避免中文 query 在英文语料上召回不全。
- 使用 Moonshot/Kimi 模型生成中文回答。

## 数据特点

数据目录：

```text
data/ML-Notes-in-Markdown-master
```

数据是英文机器学习笔记，主要特点：

- 以 Markdown 文件存储。
- 目录天然对应机器学习主题，例如 `Regression`、`Classification`、`Clustering`、`Deep Learning`。
- 文档内部包含多级标题、公式图片链接、列表和代码引用。
- 不同主题内容密度不均，有些目录只有 README，有些主题包含多个算法文件。

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

### 3. 混合检索

检索模块采用：

- FAISS：语义相似度检索。
- BM25：关键词检索，适合 `OLS`、`p-value`、`K-means` 这类精确术语。
- RRF：融合两路结果，降低单一路径漏召回风险。

整体流程：

```text
query
  -> FAISS top-k
  -> BM25 top-k
  -> RRF rerank
  -> metadata filter
  -> answer generation
```

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
project
├── code
│   ├── config.py
│   ├── main.py
│   ├── requirements.txt
│   └── rag_modules
│       ├── data_preparation.py
│       ├── generation_integration.py
│       ├── index_construction.py
│       └── retrieval_optimization.py
├── data
│   └── ML-Notes-in-Markdown-master
└── README.md
```

## 环境配置

建议使用 Python 虚拟环境：

```powershell
cd D:\1Study\0intern\rag_project\my-practice\project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
```

如果 PyPI 下载较慢，可以使用镜像：

```powershell
.\.venv\Scripts\python.exe -m pip install -r code\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

需要设置 Moonshot API key：

```powershell
$env:MOONSHOT_API_KEY="your_api_key"
```

## 运行方式

```powershell
cd D:\1Study\0intern\rag_project\my-practice\project
.\.venv\Scripts\python.exe code\main.py
```

如果 embedding 模型已经下载到本地缓存，但运行时 Hugging Face 联网检查失败，可以开启离线模式：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe code\main.py
```

首次运行会构建 FAISS 索引并保存到：

```text
code/vector_index_ml_notes
```

## Streamlit Web UI

项目也提供了一个简单的 Streamlit 界面，用于展示回答和检索来源：

```powershell
cd D:\1Study\0intern\rag_project\my-practice\project
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m streamlit run code\streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

打开浏览器访问：

```text
http://127.0.0.1:8501
```

注意：运行 Streamlit 的 PowerShell 窗口需要保持打开。关闭窗口后，本地 Web 服务会停止，浏览器会显示 connection refused。

UI 支持：

- 输入中文问题。
- 显示识别到的 query type 和 topic。
- 显示检索 query。
- 展示 top-k chunks、section path 和 source path。
- 可选择是否调用 LLM 生成最终回答。
- 在未设置 API key 时，也可以只看检索结果。

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

```powershell
cd D:\1Study\0intern\rag_project\my-practice\project
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe code\evaluate_retrieval.py --top-k 3
```

脚本会加载已有 FAISS 索引，运行内置测试集，并输出：

- Top-1 source accuracy
- Top-k source accuracy
- Top-k topic accuracy
- 每个 case 的 top-k 检索结果、topic、section 和 source path

当前基线结果：

```text
Cases: 15
Top-1 source accuracy: 40.0%
Top-3 source accuracy: 73.3%
Top-3 topic accuracy: 93.3%
```

这个结果说明：主题识别整体可用，但精确文件命中仍有优化空间，尤其是 Decision Tree、Random Forest、Gaussian Mixture 等短文档或弱关键词文档。

## 当前限制

这个项目不是完整的机器学习教材问答系统，当前限制包括：

- 数据本身内容不均衡，有些主题只有很短的 README，因此无法回答较深入问题。
- 生成质量依赖 Moonshot/Kimi API。
- 当前没有 Web UI，主要通过命令行交互。
- 离线检索评估已有初版，但测试集规模仍较小。
- metadata filter 在小主题或稀疏主题上仍可继续优化。
- 对 Markdown 中图片公式只保留链接文本，没有做 OCR 或公式解析。

## 后续计划

优先级较高的改进：

1. 增加离线检索评估脚本，统计 topic accuracy、top-1 accuracy、top-3 accuracy。
2. 增加 Streamlit Web UI，展示回答、来源文档和检索到的 chunks。
3. 展示 RRF 排名、来源路径和章节标题，提高检索可解释性。
4. 优化 metadata filter，使小主题文档不被前置检索候选数量限制。
5. 清理标题展示格式，例如 `LogisticRegression` -> `Logistic Regression`。

