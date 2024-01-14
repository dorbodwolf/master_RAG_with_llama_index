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

gemini key 配置：
```python
import os
os.environ["GOOGLE_API_KEY"] = "xxx-xxx-xxx"
```

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

#### 4.2.1 编写prompt模板
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

可以分析下这个prompt所串的流程：
```
以自然语言给出一个question {query_str}，并表征为`[query_vector]` 》》》

让pgvector以正确语法{dialect}查询数据库，数据库以{schema}来描述 》》》

以自然语言等形式返回answer
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

**深入学习：PromptTemplate类的关键贡献是什么？**

* 高级prompt技巧（变量映射、function）=> https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts.html
* prompt模板API文档 => https://docs.llamaindex.ai/en/stable/api_reference/prompts.html


#### 4.2.2 定义llm、embedding模型、table_desc、构造query

```python
print('*'*100)
# 前面已经创建数据库查询引擎engine，新建数据表sec_text_chunk并插入pdf的embedding数据
sql_database = SQLDatabase(engine, include_tables=["sec_text_chunk"])
# llm = OpenAI(model="gpt-4")
llm = Gemini()
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)
table_desc = """\
This table represents text chunks from an SEC filing. Each row contains the following columns:

id: id of row
page_label: page number 
file_name: top-level file name
text: all text chunk is here
embedding: the embeddings representing the text chunk

For most queries you should perform semantic search against the `embedding` column values, since \
that encodes the meaning of the text.

"""
# 表描述信息定义
context_query_kwargs = {"sec_text_chunk": table_desc}
# 定义查询引擎
query_engine = PGVectorSQLQueryEngine(
    sql_database=sql_database,
    text_to_sql_prompt=text_to_sql_prompt,
    service_context=service_context,
    context_query_kwargs=context_query_kwargs,
)
# 运行查询
response = query_engine.query(
    "Can you tell me about the risk factors described in page 6?",
)
print(str(response))
print('*'*100)
print(response.metadata["sql_query"])
```

输出：
```
****************************************************************************************************
I apologize, but I do not have access to external websites or specific PDF documents, including the one you cited from sec.gov. Therefore, I cannot provide you with the context from page 6 of the document.
****************************************************************************************************
SELECT text FROM sec_text_chunk WHERE page_label = 6 ORDER BY embedding <-> '[0.010814253240823746, 0.009377687238156796, 0.007271425798535347, -0.019208872690796852, 0.02425670064985752, 0.015211580321192741, 0.06400444358587265, 0.0520421527326107, -0.04742179438471794, 0.016343839466571808, 0.009889722801744938, -0.024496499449014664, -0.011254172772169113, 0.015240531414747238, -0.0273845624178648, 0.0351133830845356, 0.012452730908989906, -0.02737302891910076, -0.04164911434054375, 0.03467370197176933, -0.0007635343936271966, -0.011003962717950344, 0.0005317715695127845, -0.05463346466422081, 0.01202582661062479, 0.015133126638829708, -0.029302746057510376, 0.015582049265503883, -0.02466437965631485, -0.17029325664043427, 0.006682838778942823, -0.027216795831918716, 0.04314247518777847, -0.06312840431928635, -0.010829276405274868, 0.0034605192486196756, -0.014626051299273968, 0.040704481303691864, 0.019042454659938812, 0.00046189938439056277, -0.01629573665559292, 0.008881797082722187, 0.016320930793881416, 0.03556095436215401, 0.019315287470817566, -0.04190703481435776, -0.001360429567284882, -0.033428676426410675, 0.006985329557210207, -0.004954629577696323, -0.007235378958284855, -0.059480324387550354, -0.0003660651564132422, 0.05686912685632706, 0.05632095783948898, -0.027910245582461357, 0.03585722669959068, -0.005195773672312498, -0.03372493386268616, 0.0396389402449131, 0.025115681812167168, 0.009347704239189625, -0.21499890089035034, 0.05555267259478569, 0.01676785945892334, 0.0065127406269311905, -0.07407680153846741, 0.01000478956848383, 0.03574812039732933, 0.016660641878843307, -0.04907441884279251, 0.025313161313533783, -0.020081844180822372, 0.050977710634469986, 0.035878509283065796, -0.018191570416092873, -0.013576979748904705, -0.028686586767435074, 0.017817258834838867, 0.02599147893488407, 0.012705300003290176, 0.04637918621301651, -0.01620388962328434, -0.04377598315477371, -0.026591533794999123, -0.06055954471230507, 0.05721697956323624, -0.011904825456440449, 0.017413562163710594, 0.018094561994075775, 0.003737036371603608, -0.025049053132534027, -0.01704063080251217, 0.020439960062503815, -0.05388699844479561, -0.031089739874005318, 0.035240061581134796, 0.008267040364444256, -0.06023068726062775, 0.5355477333068848, -0.03185107558965683, -0.015646906569600105, 0.04578392952680588, -0.007183430250734091, -0.006696697324514389, -0.03661782667040825, 0.014823771081864834, -0.031056072562932968, -0.0024868010077625513, -0.0028818880673497915, 0.061232782900333405, 0.019387731328606606, 0.034376248717308044, -0.09766043722629547, 0.012555130757391453, -0.009863549843430519, 0.027941444888710976, 0.002020162297412753, 0.011386033147573471, 0.01485200971364975, -0.0020538729149848223, 0.023478375747799873, 0.009886423125863075, 0.027877943590283394, 0.030535632744431496, -0.06316938251256943, 0.012235334143042564, 0.09614087641239166, 0.03117099218070507, -0.0038573872298002243, 0.04145360365509987, -0.01940433494746685, -0.007911071181297302, 0.0194653682410717, 0.009847661480307579, -0.018310466781258583, -0.018810780718922615, 0.012774309143424034, 0.020259257405996323, 0.0003814855881500989, -0.016649771481752396, -0.05875483155250549, -0.018034284934401512, -0.08295232802629471, -0.02111334726214409, 0.06836894154548645, -4.505117976805195e-05, -0.010091242380440235, -0.021588416770100594, 0.02360152266919613, -0.014601478353142738, 0.015849871560931206, -0.0036528236232697964, -0.004234191961586475, 0.023315293714404106, -0.0017979773692786694, 0.00043892423855140805, -0.003586698090657592, -0.012089161202311516, -0.030246049165725708, -0.01838899962604046, -0.012735887430608273, -0.03640391677618027, 0.08499909937381744, 0.04863680154085159, -0.048285383731126785, -0.009110549464821815, 0.02302463725209236, -0.01045394316315651, -0.03341846540570259, 0.03876117989420891, 0.01223805733025074, -0.0037277149967849255, -0.023892784491181374, 0.052829086780548096, 0.031011158600449562, -0.015879658982157707, 0.010110381990671158, 0.009219611063599586, -0.012221292592585087, -0.007636257912963629, -0.0033193263225257397, -0.05081711336970329, 0.04637172445654869, -0.013796963728964329, -0.05341288074851036, -0.02966798096895218, -0.017490549013018608, 0.016922082751989365, 0.013800173066556454, -0.04066682234406471, -0.02067498303949833, -0.07286251336336136, 0.00758873438462615, -0.036855604499578476, 0.001179962302558124, -0.02384406328201294, -0.03992738574743271, 0.01915636844933033, -0.048795778304338455, 0.03301453962922096, 0.06913045048713684, -0.014818024821579456, 0.04127839580178261, -0.0006928873481228948, 0.04691612347960472, 0.006018996238708496, -0.04129566624760628, 0.057367097586393356, 0.023065216839313507, -0.021620839834213257, 0.017598265781998634, 0.07133748382329941, 0.0030228847172111273, -0.03860322758555412, 0.005178849212825298, 0.007941204123198986, -0.001151945674791932, 0.03912603482604027, -0.0019406439969316125, 0.0776977613568306, -0.024339066818356514, 0.002958275144919753, -0.27398931980133057, -0.0024578694719821215, 0.0025057869497686625, -0.028025628998875618, -0.011446238495409489, -0.01608985662460327, -0.007670752238482237, -0.027574067935347557, -0.0028526154346764088, 0.04913186654448509, 0.07351309806108475, 0.008516846224665642, -0.0616939477622509, 0.015419043600559235, -0.020544296130537987, -0.008283752016723156, -0.011395678855478764, -0.02402697317302227, -0.04869088530540466, 0.03601018339395523, -0.0119579266756773, 0.007755029480904341, -0.016992857679724693, 0.007297383155673742, 0.0322345569729805, 0.020029442384839058, 0.1612572818994522, -0.022014159709215164, -0.034580640494823456, -0.003593066008761525, 0.07234060764312744, 0.04575332999229431, 0.0074853901751339436, -0.08982156962156296, 0.06630075722932816, -0.015421507880091667, -0.025890015065670013, -0.01519095990806818, -0.061643604189157486, -0.03273256495594978, -0.047014299780130386, 0.002488141180947423, -0.05157428979873657, 0.01071930956095457, -0.10552021861076355, -0.024330444633960724, 0.0013028495013713837, 0.0757654532790184, -0.03166292980313301, 0.047255296260118484, 0.007477630395442247, 0.009228932671248913, 0.038919296115636826, 0.01988873817026615, 0.03487446904182434, -0.06597759574651718, -0.09367284178733826, 0.03497311845421791, -0.06651343405246735, 0.034641824662685394, -0.05199563503265381, -0.00021919574646744877, 0.08015841990709305, -0.00777824642136693, 0.04443156719207764, -0.012238739989697933, -0.013346919789910316, 0.00922362506389618, -0.02078060433268547, -0.02742745541036129, -0.06041297689080238, 0.07286405563354492, -0.026834020391106606, 0.001173133496195078, 0.03599807247519493, -0.026006188243627548, 0.034492768347263336, -0.03344705328345299, -0.040414322167634964, -0.004498907830566168, -0.008429578505456448, -0.060487017035484314, 0.008409683592617512, 0.029912520200014114, -0.005629163235425949, -0.027826650068163872, -0.02957252785563469, -0.021468063816428185, 0.022306228056550026, -0.007286489475518465, -0.027165696024894714, -0.03329389914870262, -0.010007273405790329, -0.014325127005577087, 0.024594349786639214, 0.006074496079236269, -0.2941926419734955, 0.008282206952571869, 0.015416117385029793, 0.05084770545363426, -0.018805939704179764, 0.0051071797497570515, 0.05674120411276817, 0.0023404271341860294, 0.06878603994846344, 0.030160723254084587, 0.02032998949289322, 0.02633770741522312, 0.04603448137640953, -0.0476498119533062, 0.05206853523850441, -0.0025609461590647697, 0.037075530737638474, -0.010070732794702053, 0.009689278900623322, 0.012394517660140991, 0.0416434183716774, 0.03458383306860924, 0.14666184782981873, -0.01714147999882698, -0.016475467011332512, 0.011312577873468399, 0.009565536864101887, 0.027483254671096802, -0.007222798652946949, 0.004618457984179258, 0.0400804840028286, -0.005513940006494522, 0.053462617099285126, -0.01117617916315794, 0.01243547722697258, -0.003478966886177659, -0.0026689537335187197, -0.0005776413599960506, 0.035651832818984985, -0.03519345819950104, 0.017823336645960808, 0.004927401430904865, -0.010679548606276512, -0.00785540696233511, 0.0826132521033287, -0.009108821861445904, -0.03728794306516647, -0.08854090422391891, 0.03262175992131233, 0.002886422211304307, -0.01665397919714451, -0.018871404230594635, 0.013413931243121624, -0.0017775617307052016, 0.027674630284309387, 0.022784892469644547, 0.016291562467813492, -0.023320043459534645, 0.008166699670255184, 0.004573751240968704, 0.01589772291481495, -0.03184141218662262, -0.02086031064391136, 0.04537849500775337, 0.00260591390542686]' LIMIT 5
```


再次看prompt里面的回答模板：
```
    Question: Question here\n
    SQLQuery: SQL Query to run\n
    SQLResult: Result of the SQLQuery\n
    Answer: Final answer here\n\n
```

但是看返回的答案`Answer`似乎不如人意，分析如下：

以上输出的这串向量就是prompt模板里面的`Question`的`query_vector`。

看样子查询语句`SQLQuery`是构造成功了，

是否查询返回了查询结果即`SQLResult`？

还是因为gemini没有理解`SQLResult`？


打印出来response对象（`<class 'llama_index.response.schema.Response'>`）的metadata里面的result的内容，应该就是SQLResult：

```python
(Pdb) print(response.metadata['result'])
[('Impact of COVID-19 to our BusinessThe\n ongoing  COVID-19  pandemic  continues  to  impact  communities  in  the  United  States,  Canada  and  globa ... (2741 characters truncated) ... r Transportation Network\nOur transportation network off\ners riders seamless, personalized and on-demand access to a variety of mobility options.\n6',)]
```

#### 结论

对比了gemini、llama-2-13b、vicuna-13b、gpt-3.5-turbo，只有gpt能够返回符合预期的提取答案。

## TODO llamaindex在LinkedIn发的一些有用项目

### Q&A
### chatbots
### agents
### 多模态