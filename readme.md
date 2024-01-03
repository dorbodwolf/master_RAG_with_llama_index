## 1. 写在前面
借助llama-index库来实践基于LLM的应用
ref: https://docs.llamaindex.ai/en/stable/index.html


## 2. 基于gemini和bge来完成llama_index的Starter Tutorial
ref: https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html

**LLM API选择**

llama_index默认使用openAI api，对于快速上手来说最合适的就是openai api。
* 私有部署小模型：没有GPU资源
* 购买openai API配额：信用卡死活不通过
* 最终选择Google gemini pro：llama_index支持gemini调用（本来以为不支持，配置openai proxy也可以通过模仿openai api来调用）

**embedding模型选择**

llama_index默认调用openAI的embedding模型，但是可以选择本地模型（实际是调用HuggingFaceEmbedding），默认是```BAAI/bge-small-en```

```python
# llm选择和embedding模型选择
service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')
```
logging的debug日志可以看到与huggingface仓库的交互：
```
DEBUG:urllib3.connectionpool:Starting new HTTPS connection (1): huggingface.co:443
Starting new HTTPS connection (1): huggingface.co:443
DEBUG:urllib3.connectionpool:https://huggingface.co:443 "HEAD /BAAI/bge-small-en/resolve/main/config.json HTTP/1.1" 200 0
https://huggingface.co:443 "HEAD /BAAI/bge-small-en/resolve/main/config.json HTTP/1.1" 200 0
DEBUG:urllib3.connectionpool:https://huggingface.co:443 "HEAD /BAAI/bge-small-en/resolve/main/tokenizer_config.json HTTP/1.1" 200 0
https://huggingface.co:443 "HEAD /BAAI/bge-small-en/resolve/main/tokenizer_config.json HTTP/1.1" 200 0
```

**加载本地文档**

```Python
documents = SimpleDirectoryReader(data_dir).load_data()
```
日志：
```
DEBUG:llama_index.readers.file.base:> [SimpleDirectoryReader] Total files added: 1
> [SimpleDirectoryReader] Total files added: 1
```

**对本地文档做chunk操作**

```python
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
```
Vector Stores是retrieval-augmented generation(RAG)的关键组成部分。当调用from_documents，文件会被分割为chunks并解析为Node objects,这是对文本字符串的轻量化抽象，来保持元数据和关系。

这种方式默认在内存中存储vector embedding，稍后会存下来方便后续加载（节省时间和embedding调用花销）。

日志：
```
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: What I Worked On

February 2021

Before college...
> Adding chunk: What I Worked On

February 2021

Before college...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: I couldn't have put this into words when I was ...
> Adding chunk: I couldn't have put this into words when I was ...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: Over the next several years I wrote lots of ess...
> Adding chunk: Over the next several years I wrote lots of ess...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: So we just made what seemed like the obvious ch...
> Adding chunk: So we just made what seemed like the obvious ch...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: I don't think it was entirely luck that the fir...
> Adding chunk: I don't think it was entirely luck that the fir...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: YC was different from other kinds of work I've ...
> Adding chunk: YC was different from other kinds of work I've ...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: For the rest of 2013 I left running YC more and...
> Adding chunk: For the rest of 2013 I left running YC more and...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: Now they are, though. Now you could continue us...
> Adding chunk: Now they are, though. Now you could continue us...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: Notes

[1] My experience skipped a step in the ...
> Adding chunk: Notes

[1] My experience skipped a step in the ...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: Customary VC practice had once, like the custom...
> Adding chunk: Customary VC practice had once, like the custom...
```

**基于index实例化query_engine**

```python
query_engine = index.as_query_engine(service_context=service_context)
```
方法定义如下：
```
    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        # NOTE: lazy import
        from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

        retriever = self.as_retriever(**kwargs)

        kwargs["retriever"] = retriever
        if "service_context" not in kwargs:
            kwargs["service_context"] = self._service_context
        return RetrieverQueryEngine.from_args(**kwargs)
```
其中的as_retriever方法实例化了一个VectorIndexRetriever对象：
```python
class VectorIndexRetriever(BaseRetriever):
    """Vector index retriever.

    Args:
        index (VectorStoreIndex): vector store index.
        similarity_top_k (int): number of top k results to return.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        alpha (float): weight for sparse/dense retrieval, only used for
            hybrid query mode.
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

    """
```

**query执行**

```python
response = query_engine.query("What did the author do growing up?")
print('Answer: ', response)
```
调用的方法：
```python
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._guideline_evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryGuidelineQueryEngine(
                self._query_engine,
                self._guideline_evaluator,
                self.resynthesize_query,
                self.max_retries - 1,
                self.callback_manager,
            )
            new_query = self.query_transformer.run(query_bundle, {"evaluation": eval})
            logger.debug("New query: %s", new_query.query_str)
            return new_query_engine.query(new_query)
```

日志：
```
DEBUG:llama_index.indices.utils:> Top 2 nodes:
> [Node 473dffaf-d857-4bc2-8a01-6b0d92112ba7] [Similarity score:             0.806073] What I Worked On

February 2021

Before college the two main things I worked on, outside of schoo...
> [Node dd7262a5-e922-48dc-8907-48641e27f1c9] [Similarity score:             0.784516] I couldn't have put this into words when I was 18. All I knew at the time was that I kept taking ...

```

最终输出答案：
```
Answer: Growing up, the author worked on writing and programming. He wrote short stories and tried writing programs on an IBM 1401 computer. Later, he got a microcomputer and started programming more seriously, writing simple games, a program to predict how high his model rockets would fly, and a word processor that his father used to write a book.
```


完整代码：
```python
import os
os.environ["GOOGLE_API_KEY"] = "xxx-xxx-xxx"
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms import Gemini

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # 比较详细的logging等级，方便学习
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

data_dir = '../data'

service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')

documents = SimpleDirectoryReader(data_dir).load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What did the author do growing up?")
print('Answer: ', response)
```

**存储index内容**

只需一行代码：
```python
index.storage_context.persist(persist_dir = '../data')
```

可以看到存储到了制定文件夹:
```bash
» ls -lh ../data
total 824
-rw-r--r--@ 1 jade_mayer  staff   187K  1  1 23:41 default__vector_store.json
-rw-r--r--@ 1 jade_mayer  staff   135K  1  1 23:41 docstore.json
-rw-r--r--@ 1 jade_mayer  staff    18B  1  1 23:41 graph_store.json
-rw-r--r--@ 1 jade_mayer  staff    72B  1  1 23:41 image__vector_store.json
-rw-r--r--@ 1 jade_mayer  staff   2.0K  1  1 23:41 index_store.json
-rw-r--r--@ 1 jade_mayer  staff    73K  1  1 22:18 paul_graham_essay.txt
```
embedding向量在default__vector_store.json里面，可以看到每个token的输出都是384维，正是```BAAI/bge-small-en```的输出维度

ref： https://huggingface.co/BAAI/bge-small-en

```
{
    "embedding_dict": {
        "32ba1f7f-3ef5-4118-b3ce-172b4f127168": [
            -0.04698172211647034,
            -0.015481043606996536,
            0.023415159434080124,
            // ...
            0.01598987728357315,
            0.006660557352006435,
            -0.026917191222310066,
            -0.0028073545545339584,
            -0.014330701902508736,
            -0.03930317237973213,
            0.02456594631075859,
            3,
            0.04485808312892914,
            0.02181057073175907
        ],
        // ...
    },
    "text_id_to_ref_doc_id": {
        "32ba1f7f-3ef5-4118-b3ce-172b4f127168": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "65d0c3f3-b2a9-4bb4-91be-ad42987529b5": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "5c235a2d-7113-48fd-bde0-13b8753a0d1f": "93d53b99-545d-466f-a1e0-50a62ab22951",
        // ...
        "6a7a062b-7086-4e45-a5c2-d95e8e1e28f9": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "973527a1-ceac-4c5a-b9e8-b17f842c3d44": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "48b7e978-dbb8-4764-9ce8-91979e03e8a8": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "063cbd8a-22a3-4d44-93d0-a8509005f5bf": "93d53b99-545d-466f-a1e0-50a62ab22951"
    },
    "metadata_dict": {
        "32ba1f7f-3ef5-4118-b3ce-172b4f127168": {
            "file_path": "../data/paul_graham_essay.txt",
            "file_name": "paul_graham_essay.txt",
            "file_type": "text/plain",
            "file_size": 75042,
            "creation_date": "2024-01-01",
            "last_modified_date": "2024-01-01",
            "last_accessed_date": "2024-01-01",
            "_node_type": "TextNode",
            "document_id": "93d53b99-545d-466f-a1e0-50a62ab22951",
            "doc_id": "93d53b99-545d-466f-a1e0-50a62ab22951",
            "ref_doc_id": "93d53b99-545d-466f-a1e0-50a62ab22951"
        },
        // ...
    }
}
```

直接加载存储好的文档embedding来重复这个任务：
```python
import os
os.environ["GOOGLE_API_KEY"] = "xxx-xxx-xxx"
# from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms import Gemini

# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # 比较详细的logging等级，方便学习
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# data_dir = '../data'

# service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')

# documents = SimpleDirectoryReader(data_dir).load_data()
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# query_engine = index.as_query_engine(service_context=service_context)
# response = query_engine.query("What did the author do growing up?")
# print('Answer: ', response)

# index.storage_context.persist(persist_dir = '/Users/jade_mayer/projects/agents/llamaindex/data')

import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')
# check if storage already exists
PERSIST_DIR = "../data"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("../data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)

# either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```

## 3. VectorStore自定义

llama_index默认的向量库是在mem里面构建字典结构的embedding存储。见```llama_index/llama_index/vector_stores/simple.py```

除了Meta的FAISS（https://github.com/facebookresearch/faiss ），llama_index中许多vector stores同时存储数据和index(embeddings)。

这里按照官方示例跑通chroma的demo，和chroma相关的变更在下面代码中加以注释

```python
import os
os.environ["GOOGLE_API_KEY"] = "xxx-xxx-xxx"
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms import Gemini

import logging
import sys

# 导入chromadb包
import chromadb
# 导入llama_index的ChromaVectorStore类
from llama_index.vector_stores import ChromaVectorStore
from llama_index import StorageContext


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # 比较详细的logging等级，方便学习
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

data_dir = '../data'

service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')

documents = SimpleDirectoryReader(data_dir).load_data()

# 创建chromadb的连接和存储
chroma_client = chromadb.PersistentClient()
chroma_collection = chroma_client.create_collection("deyusdemo")
# 创建ChromaVectorStore实例
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# 基于chroma自定义StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 相比原有的index构造新增自定义StorageContext，替换为chroma的向量存储
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What did the author do growing up?")
print('Answer: ', response)
```

chroma在```master_RAG_with_llama_index/chroma```下面创建了数据存储。

## 4. Text-to-SQL with PGVector
https://docs.llamaindex.ai/en/stable/examples/query_engine/pgvector_sql_query_engine.html

测试数据："Lyft 2021 10k document" Lyft 公司在 2021 年提交给美国证券交易委员会 (U.S. Securities and Exchange Commission, SEC) 的 10-K 表格。10-K 表格是美国上市公司每年按照 SEC 规定提交的年度报告，其中包含了公司的财务状况、运营结果、管理层讨论与分析等重要信息。

docker run postgres
```bash
docker run --hostname=0c4896b7ea54 --mac-address=02:42:ac:11:00:02 --env=POSTGRES_USER=jade_mayer --env=POSTGRES_DB=my_database  --env=POSTGRES_PASSWORD=123456 --env=POSTGRES_HOST_AUTH_METHOD=trust --env=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/postgresql/15/bin --env=GOSU_VERSION=1.16 --env=LANG=en_US.utf8 --env=PG_MAJOR=15 --env=PG_VERSION=15.5-1.pgdg110+1 --env=PGDATA=/var/lib/postgresql/data --volume=/var/lib/postgresql/data -p 5432:5432 --runtime=runc -d postgres:15.5-bullseye
```
官方的postgres docker image 没有安装vector扩展，可以参考这个： https://github.com/pgvector/pgvector?tab=readme-ov-file#docker

```bash
docker run --hostname=880a19bb03a2 --env=POSTGRES_USER=jade_mayer --env=POSTGRES_DB=my_database  --env=POSTGRES_PASSWORD=123456 --env=POSTGRES_HOST_AUTH_METHOD=trust  -p 5432:5432 --env=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/postgresql/15/bin --env=GOSU_VERSION=1.16 --env=LANG=en_US.utf8 --env=PG_MAJOR=15 --env=PG_VERSION=15.4-2.pgdg120+1 --env=PGDATA=/var/lib/postgresql/data --volume=/var/lib/postgresql/data --runtime=runc -d ankane/pgvector:latest
```

### 4.1 embedding插入pgvector库

基于启动的pgvector数据库服务来建立连接：
```python
engine = create_engine("postgresql+psycopg2://jade_mayer:123456@localhost:5432/my_database")
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
```

将文本及bge模型提取的embedding插入pgvector：

```python
# 定义表schema，存储page_label embedding text
Base = declarative_base()
class SECTextChunk(Base):
    __tablename__ = "sec_text_chunk"
    id = mapped_column(Integer, primary_key=True)
    page_label = mapped_column(Integer)
    file_name = mapped_column(String)
    text = mapped_column(String)
    embedding = mapped_column(Vector(384))

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# get embeddings for each node
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
for node in nodes:
    text_embedding = embed_model.get_text_embedding(node.get_content())
    node.embedding = text_embedding

# insert into database
for node in nodes:
    row_dict = {
        "text": node.get_content(),
        "embedding": node.embedding,
        **node.metadata,
    }
    stmt = insert(SECTextChunk).values(**row_dict)
    with engine.connect() as connection:
        cursor = connection.execute(stmt)
        connection.commit()
```

### 4.2 定义PGVectorSQLQueryEngine
插入数据后配置查询引擎。改进默认的text-to-sql prompt来适配pgvector的语法，并通过一些few-shot例子来prompt正确使用语法。

下面是一个prompt模板：
```python
text_to_sql_tmpl = """\
Given an input question, first create a syntactically correct {dialect} \
query to run, then look at the results of the query and return the answer. \
You can order the results by a relevant column to return the most \
interesting examples in the database.

Pay attention to use only the column names that you can see in the schema \
description. Be careful to not query for columns that do not exist. \
Pay attention to which column is in which table. Also, qualify column names \
with the table name when needed. 

IMPORTANT NOTE: you can use specialized pgvector syntax (`<->`) to do nearest \
neighbors/semantic search to a given vector from an embeddings column in the table. \
The embeddings value for a given row typically represents the semantic meaning of that row. \
The vector represents an embedding representation \
of the question, given below. Do NOT fill in the vector values directly, but rather specify a \
`[query_vector]` placeholder. For instance, some select statement examples below \
(the name of the embeddings column is `embedding`):
SELECT * FROM items ORDER BY embedding <-> '[query_vector]' LIMIT 5;
SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5;
SELECT * FROM items WHERE embedding <-> '[query_vector]' < 5;

You are required to use the following format, \
each taking one line:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use tables listed below.
{schema}


Question: {query_str}
SQLQuery: \
"""
```

基于这个prompt模板来构造prompt
```python
text_to_sql_prompt = PromptTemplate(text_to_sql_tmpl)
```
输出的text_to_sql_prompt是这样的：
```
metadata={'prompt_type': <PromptType.CUSTOM: 'custom'>} 
template_vars=['dialect', 'schema', 'query_str'] 
kwargs={} output_parser=None 
template_var_mappings=None 
function_mappings=None 

template=
"
Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.\n\n
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. \n\n
IMPORTANT NOTE: you can use specialized pgvector syntax (`<->`) to do nearest neighbors/semantic search to a given vector from an embeddings column in the table. The embeddings value for a given row typically represents the semantic meaning of that row. 
The vector represents an embedding representation of the question, given below. Do NOT fill in the vector values directly, but rather specify a `[query_vector]` placeholder. 
For instance, some select statement examples below (the name of the embeddings column is `embedding`):\n
    SELECT * FROM items ORDER BY embedding <-> '[query_vector]' LIMIT 5;\n
    SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5;\n
    SELECT * FROM items WHERE embedding <-> '[query_vector]' < 5;\n\n
You are required to use the following format, each taking one line:\n\n
    Question: Question here\n
    SQLQuery: SQL Query to run\n
    SQLResult: Result of the SQLQuery\n
    Answer: Final answer here\n\n
Only use tables listed below.\n
    {schema}\n\n\n

Question: {query_str}\n
SQLQuery: 
"
```

**思考：PromptTemplate类的关键贡献是什么？**

## TODO llamaindex在LinkedIn发的一些有用项目

### Q&A
### chatbots
### agents
### 多模态