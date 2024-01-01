import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyC2G-gs329x0LYVvoCGqCw23sQR2FKq-LE"
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