# ML Notes RAG 项目面试核心问答

这份材料不按题库死背，而是按面试追问链路整理。每组准备 3-5 个核心问题，回答时要能从“为什么做”讲到“怎么做”“有什么问题”“怎么优化”。

## 一、项目动机类

### 1. 你用 1 分钟介绍一下这个项目。

答：这是一个面向英文机器学习 Markdown 笔记的中文 RAG 问答系统。项目把 `ML-Notes-in-Markdown` 这批英文机器学习资料处理成可检索知识库，用户用中文提问时，系统会先做问题分类和跨语言 query rewrite，再用 FAISS 向量检索、BM25 关键词检索和 RRF 融合重排找到相关 chunk，最后调用 Moonshot/Kimi 基于检索上下文生成中文回答，并展示来源路径和章节。它的重点不是单纯做聊天，而是让答案尽量来自指定笔记，并且可追溯。

### 2. 为什么这个项目适合用 RAG，而不是直接让 LLM 回答？

答：因为这个项目的目标是回答“这批机器学习笔记里有什么”，而不是让模型凭通用知识回答。直接问 LLM 可能会答得很流畅，但它不一定来自项目资料，也无法展示来源。RAG 的价值在于先检索指定知识库，再让 LLM 基于检索结果回答，这样能降低幻觉，也方便用户验证答案依据。

### 3. 用户提问后，系统完整链路是什么？

答：用户输入中文问题后，系统先判断问题类型，比如列表类、详细解释类或一般问题；然后根据中文 alias 识别 topic；如果是列表类问题且 topic 明确，就直接按 topic 列出文档；否则会把中文问题改写成适合英文语料检索的 query，再执行 FAISS + BM25 混合检索，用 RRF 融合重排，取 top-k chunk 构建上下文，最后交给 LLM 生成中文回答。

### 4. 这个项目最核心的难点是什么？

答：核心难点是中文问题检索英文资料。BM25 对中文 query 和英文文档天然不匹配，向量检索也依赖 embedding 模型的跨语言对齐能力。所以项目用了多语言 embedding、中文 topic alias、LLM query rewrite 和 fallback rewrite 来提升跨语言召回。

### 5. 这个项目目前最大的不足是什么？

答：最大不足是精确 source 命中率还不高。README 里记录的 Top-1 source accuracy 是 40%，Top-3 source accuracy 是 73.3%，说明系统大多能找到正确主题，但第一名不一定是最准确文件。后续需要优化 chunk 表示、query rewrite、metadata filtering 和 reranking。

## 二、文档处理类

### 1. 数据来源和格式是什么？

答：数据来自 `ML-Notes-in-Markdown`，是一批英文 Markdown 机器学习笔记。目录按主题组织，比如 Regression、Classification、Clustering、Deep Learning、Time Series 等。Markdown 本身有标题层级、列表、代码片段和图片公式链接，所以很适合做结构化解析。

### 2. 为什么要专门处理 Markdown 标题结构？

答：Markdown 标题天然表达语义层级。比如 `Linear Regression > Variable Selection` 就是一个清晰的小节。如果只按固定字符数切分，可能把一个概念拆断，也可能把多个不相关章节混在一起。按标题切分能让 chunk 更完整、更可解释，来源展示也更清楚。

### 3. 你们保留了哪些 metadata？

答：主要保留 `title`、`topic`、`chapter`、`relative_path`、`source`、`section_path`、`parent_id`、`chunk_id`、`chunk_index`、`chunk_size` 等。`topic` 用于过滤和列表问题，`section_path` 用于定位章节，`relative_path` 用于来源展示和评估，`parent_id` 用于从 chunk 回溯到原文档。

### 4. topic 是怎么识别的？

答：主要从一级目录名提取，比如 `01-Regression` 去掉数字前缀后变成 `Regression`。同时项目维护了 `TOPIC_ALIASES`，用中文别名匹配 topic，比如“回归”“线性回归”匹配到 Regression，“聚类”“K-means”匹配到 Clustering。

### 5. 图片公式现在怎么处理？有什么改进空间？

答：当前图片公式主要保留为 Markdown 链接文本，没有做 OCR 或公式解析。改进时可以在预处理阶段解析图片路径，用 OCR 或公式识别模型提取 LaTeX/文本，再把识别结果加入对应 chunk，重新构建索引。

## 三、Chunk 类

### 1. chunk 是怎么生成的？

答：项目先用 Markdown header splitter 按 `#`、`##`、`###`、`####` 切分，得到标题感知的片段。如果某个标题块过长，再用递归字符切分器按长度继续切，默认 chunk size 约 1200，overlap 约 150。

### 2. chunk size 和 chunk overlap 分别影响什么？

答：chunk size 决定单个片段承载的信息量。太大会带来噪声、占用上下文；太小会缺少完整解释。overlap 用于保留切分边界附近的信息，避免关键句子被截断后上下文丢失。

### 3. 为什么要区分 parent document 和 child chunk？

答：检索时适合用 child chunk，因为粒度小、相关性更强；展示列表或回溯来源时适合用 parent document，因为用户需要知道原始文档是什么。这个设计能兼顾检索精度和来源可解释性。

### 4. 当前 chunk 策略有什么问题？

答：短文档和弱关键词文档容易召回不稳定，因为它们提供给 embedding 和 BM25 的信息少，排名容易被长文档压过。另外，当前 chunk embedding 主要基于 chunk 内容，如果没有把 title、topic、section_path 拼进去，短 chunk 的主题信息可能不足。

### 5. 你会怎么优化 chunk？

答：我会先用评估集比较不同 chunk size 和 overlap，然后把 title、topic、section_path 拼到 chunk 文本前参与 embedding；对短文档做 heading boosting；必要时引入 parent-child retriever，用小 chunk 检索、大文档补上下文。

## 四、Embedding / FAISS 类

### 1. embedding 在项目里做什么？

答：embedding 把用户 query 和文档 chunk 转成向量，让语义相近的文本在向量空间里距离更近。这样即使中文问题和英文资料字面不同，也有机会通过语义相似度召回相关内容。

### 2. 为什么选择 multilingual embedding？

答：因为用户主要用中文提问，而知识库是英文 Markdown。多语言 embedding 可以把中文和英文映射到同一个语义空间，否则中文 query 很难召回英文 chunk。

### 3. FAISS 在这里的作用是什么？

答：FAISS 是本地向量索引，用于高效执行相似度搜索。项目首次运行时会根据 chunks 构建 FAISS index 并保存，后续启动优先加载本地索引，避免重复 embedding 和建索引。

### 4. 为什么要保存 FAISS index？

答：embedding 计算和索引构建比较耗时，保存索引可以提升启动速度。对于本地演示项目，这也能避免每次运行都重新下载模型或重新计算向量。

### 5. 向量检索有什么局限？

答：向量检索适合语义相似问题，但不擅长严格字面匹配，比如 `p-value`、`OLS`、`PCA`、路径、代码符号等。它也依赖 embedding 模型质量，如果中英对齐不好，跨语言召回会下降。

## 五、BM25 / RRF 类

### 1. 为什么项目同时用了 FAISS 和 BM25？

答：因为两者互补。FAISS 擅长语义召回，适合“变量选择怎么做”这类自然语言问题；BM25 擅长关键词匹配，适合 `K-means`、`p-value`、`Naive Bayes`、`PCA` 这类精确术语。混合检索能降低单一路径漏召回的风险。

### 2. BM25 和向量检索的核心区别是什么？

答：BM25 看词面匹配，基于词频、逆文档频率和文档长度；向量检索看语义相似，基于 embedding 后的向量距离。BM25 更可解释，向量检索更能处理同义表达和跨语言语义。

### 3. RRF 是什么？为什么用它融合结果？

答：RRF 是 Reciprocal Rank Fusion，用排名位置融合多个检索器结果，公式大致是 `score += 1 / (k + rank)`。它不直接比较 FAISS 和 BM25 的原始分数，因为两者分数尺度不同，只用排名融合更稳。

### 4. RRF 里的 `k=60` 有什么作用？

答：`k` 是平滑参数，避免排名靠前结果分数过度拉开。`k` 越大，不同名次之间的分数差异越平缓，融合结果更稳定。

### 5. 当前 RRF 实现有什么潜在问题？

答：当前用 `hash(doc.page_content)` 作为文档 ID，如果两个 chunk 内容相同但来源不同，可能被误合并，导致来源丢失。更稳的做法是使用 `chunk_id` 或 `relative_path + section_path + chunk_index` 作为唯一 ID。

## 六、Query Rewrite 类

### 1. 为什么需要 query rewrite？

答：因为用户是中文问题，知识库是英文文档。直接用中文检索英文 Markdown，BM25 基本匹配不到，向量检索也可能不稳定。query rewrite 把中文问题改写成英文机器学习术语，能明显提升英文资料召回。

### 2. query rewrite 怎么实现？

答：详细问题会调用 LLM，根据 prompt 生成适合检索英文笔记的短 query。比如“线性回归怎么做变量选择？”可以改写成 `linear regression variable selection feature selection p-value backward elimination`。

### 3. query rewrite 会带来什么风险？

答：它可能把问题改偏，或者补充错误关键词。一旦 query rewrite 出错，FAISS 和 BM25 都会被带偏，最终 LLM 拿到的上下文也会错。所以需要展示 retrieval query，并配合 fallback rewrite 和评估集验证效果。

### 4. 列表类问题为什么不一定走 query rewrite + 检索？

答：列表类问题常常是在问某个主题下有哪些内容，比如“分类有哪些方法”。这种情况下直接根据 topic metadata 列出 `Classification` 下的文档，比向量检索更完整、更稳定。

### 5. 如何让跨语言检索更稳定？

答：可以扩充中文 alias，优化 LLM rewrite prompt，加入 query expansion，使用更强多语言 embedding，把 title/topic/section 拼进 embedding 文本，并引入 reranker 做精排。

## 七、评估指标类

### 1. 为什么要做离线检索评估？

答：RAG 的答案质量首先取决于检索是否找到正确资料。离线评估不调用 LLM，只评估检索模块，可以排除 LLM 随机性、API 成本和生成风格的影响，更清楚地看到召回问题。

### 2. `evaluate_retrieval.py` 评估什么？

答：它内置了一组 EvalCase，每个 case 包括用户问题、检索 query、预期 topic 和预期 source path。脚本运行 hybrid search 后，检查 top-k 结果是否命中预期主题和源文件。

### 3. Top-1 source accuracy、Top-k source accuracy、Top-k topic accuracy 分别是什么？

答：Top-1 source accuracy 看第一条结果是否来自预期文件；Top-k source accuracy 看前 k 条里是否有预期文件；Top-k topic accuracy 看前 k 条里是否有预期主题。source 更细，topic 更粗。

### 4. README 里的指标说明了什么？

答：当前 15 个 case 上 Top-1 source accuracy 是 40%，Top-3 source accuracy 是 73.3%，Top-3 topic accuracy 是 93.3%。这说明系统大多数时候能找到正确主题，但第一名排序和具体文件命中还不稳定。

### 5. 如果扩展评估集，你会怎么设计？

答：每个主题至少设计多个中文问题，覆盖列表、概念解释、步骤、公式、优缺点、代码术语和边界表达。每个 case 标注 expected topic、source path，最好再标注 expected section，用来更细粒度评估。

## 八、失败 Case 类

### 1. 这个项目里哪些 case 容易失败？

答：短文档、弱关键词文档和主题相近文档容易失败，比如 Decision Tree、Random Forest、Gaussian Mixture 这类内容较短或术语重叠的文档，可能被更长、更丰富的相关文档压过。

### 2. 为什么短文档容易召回不稳定？

答：短文档可供 embedding 学习的上下文少，BM25 可匹配词也少。如果 query 中的关键词也不完整，短文档很难在排序中超过内容更丰富的文档。

### 3. 如果 topic 匹配错了，会怎样？

答：metadata filter 会把正确结果过滤掉。比如用户表达没有覆盖在 alias 中，或者误匹配到别的 topic，后续即使混合检索找到相关 chunk，也可能被过滤掉。

### 4. query rewrite 失败会怎样？

答：如果 rewrite 把问题改偏，检索会召回错误上下文。LLM 可能基于错误上下文生成看似合理但不相关的答案，所以 UI 中展示 retrieval query 很重要。

### 5. 你会怎么定位失败原因？

答：我会记录原始问题、rewrite query、识别 topic、FAISS top-k、BM25 top-k、RRF 后 top-k、source path 和 section。先判断是 query rewrite 问题、topic filter 问题、embedding 问题、BM25 问题，还是 RRF 排序问题。

## 九、如果上线真实业务类

### 1. 这个项目能直接上线吗？

答：不能直接上线。它更像本地演示和原型，缺少鉴权、并发控制、日志监控、错误追踪、索引版本管理、API key 安全管理和稳定部署方案。

### 2. 上线前你会补哪些能力？

答：我会补统一服务层、用户鉴权、密钥管理、结构化日志、监控告警、索引 manifest、缓存失效机制、错误重试、限流和用户反馈收集。

### 3. 如何管理 API key？

答：API key 应该放在环境变量、部署平台 secret 或密钥管理服务中，不能写入代码或提交到仓库。生产环境还要限制权限、做轮换和访问审计。

### 4. 如何降低 LLM 幻觉？

答：首先保证检索质量；其次 prompt 中要求只能依据上下文回答；检索不到时明确说依据不足；答案展示来源；还可以做 groundedness 检查，评估答案中的事实是否能在引用 chunk 中找到。

### 5. 如何防 prompt injection？

答：文档内容进入 LLM context 时，可能包含恶意指令。可以在 prompt 中明确文档只是资料不是指令，对文档内容做边界标记和清洗，必要时加入注入检测，并避免把系统密钥或内部信息暴露给模型。

## 十、如果数据量扩大类

### 1. 如果数据量扩大 100 倍，哪里会成为瓶颈？

答：embedding 计算、FAISS 建索引、BM25 内存占用、启动加载时间、Streamlit 单进程并发和 LLM 上下文成本都会成为瓶颈。原型里的同步本地流程需要改成更工程化的索引构建和服务化部署。

### 2. 如何处理索引更新？

答：需要引入索引 manifest，记录文件 hash、mtime、chunk 参数、embedding 模型版本和索引版本。启动或构建时比较 manifest，判断哪些文档新增、删除或修改，再决定增量更新或全量重建。

### 3. 当前是否支持增量索引？

答：代码里有 `add_documents` 方法，但主流程没有完整实现增量检测和删除更新。真实业务里需要支持新增、修改、删除文档的索引同步。

### 4. metadata filter 先检索后过滤有什么问题？

答：如果正确 topic 的文档没有进入初始候选集，后置过滤会漏召回。数据量变大后，这个问题会更明显。更好的方式是检索前按 metadata 缩小候选集合，或者为不同 topic 建独立索引。

### 5. 大规模场景下你会怎么改架构？

答：我会把数据处理、索引构建、检索服务和生成服务拆开；向量库换成支持 metadata filter 和增量更新的服务；BM25 用专门搜索引擎；加入 reranker；日志和评估数据入库；前端只调用后端 API。

## 十一、工程质量与重构类

### 1. 当前模块划分是否合理？

答：整体合理，分成数据准备、索引构建、检索优化、生成集成、CLI 和 Streamlit UI，职责比较清楚。但 CLI 和 UI 之间有部分逻辑重复，后续应该抽象统一服务层。

### 2. CLI 和 Streamlit 的逻辑有什么不一致？

答：CLI 更依赖 LLM 做 router 和 rewrite，并且初始化时强制要求 `MOONSHOT_API_KEY`；Streamlit 有本地 heuristic router 和 fallback rewrite，还允许关闭 LLM 只看检索。这会导致同一个问题在两个入口结果不一致。

### 3. 你会如何重构？

答：我会抽象一个 `RAGService`，统一封装 route、rewrite、retrieve、generate、fallback 和 source formatting。CLI 和 Streamlit 只负责输入输出，避免业务逻辑分叉。

### 4. 当前安全上有什么注意点？

答：`FAISS.load_local(..., allow_dangerous_deserialization=True)` 有反序列化风险，只能加载自己生成且可信的索引。API key 也必须通过环境变量管理，不能写进代码。

### 5. 如果继续做两周，优先级是什么？

答：第一是扩展评估集；第二是优化短文档召回和 query rewrite；第三是统一 CLI/UI 核心逻辑；第四是引入 reranker；第五是加索引版本管理和失败 case 日志。

## 十二、项目表达类

### 1. 这个项目最能体现你的什么能力？

答：它体现了我对 RAG 全链路的理解，包括 Markdown 数据处理、chunk 策略、metadata 设计、embedding、FAISS、BM25、RRF、query rewrite、LLM 生成、Streamlit 展示和离线评估。

### 2. 如果面试官说“这不就是调库吗”，你怎么回应？

答：RAG 的难点不只是调库，而是把数据处理、检索策略和评估闭环做好。比如怎么切 chunk、怎么设计 metadata、怎么解决中文 query 检索英文文档、怎么融合 FAISS 和 BM25、怎么评估 source accuracy，这些都会直接影响系统质量。

### 3. 你遇到过最大的困难是什么？

答：最大困难是跨语言检索不稳定，特别是短文档和弱关键词文档。为了解决这个问题，我加入了中文到英文 query rewrite、多语言 embedding、topic alias、混合检索和离线评估，用数据定位召回问题。

### 4. 你怎么验证项目有效？

答：除了人工提问测试，我写了离线检索评估脚本，不调用 LLM，只看检索是否命中预期 topic 和 source path。这样可以量化 Top-1、Top-k source accuracy 和 topic accuracy，发现具体失败 case。

### 5. 你从这个项目学到了什么？

答：我学到 RAG 的效果很大程度不取决于最后的 LLM，而取决于前面的数据结构化、chunk、metadata、query rewrite 和检索评估。尤其在跨语言场景下，检索链路是否稳定比 prompt 写得漂亮更关键。
