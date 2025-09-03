import os
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:12334'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:12334'

# 定义持久化路径
persist_directory = '../../../vector_db/chroma'

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

from langchain_community.vectorstores import Chroma
from file_splitter import split_pdf


def generate_docs():
    return split_pdf()


split_docs = generate_docs();

# 使用 OpenAI Embedding
from langchain_community.embeddings import OpenAIEmbeddings
def generate_embedding():
    return OpenAIEmbeddings()


embedding = generate_embedding();



vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

print(f"向量库中存储的数量：{vectordb._collection.count()}")

question="什么是大语言模型"

# 如果只考虑检索出内容的相关性会导致内容过于单一，可能丢失重要信息。
sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")

for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

#最大边际相关性 (MMR, Maximum marginal relevance) 可以帮助我们在保持相关性的同时，增加内容的丰富度。
#核心思想是在已经选择了一个相关性高的文档之后，再选择一个与已选文档相关性较低但是信息丰富的文档。这样可以在保持相关性的同时，增加内容的多样性，避免过于单一的结果。
mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
