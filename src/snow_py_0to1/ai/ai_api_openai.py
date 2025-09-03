import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，还需要做如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:12334'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:12334'

################################################################
###  call OpenAI api
################################################################
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 导入所需库
# 注意，此处我们假设你已根据上文配置了 OpenAI API Key，如没有将访问失败
completion = client.chat.completions.create(
    # 调用模型：ChatGPT-4o
    model="gpt-4o-mini",
    # messages 是对话列表
    messages=[
        {"role": "system", "content": "你是一个幽默的对我非常有帮助的助手."},
        {"role": "user", "content": "今天心情好吗!"}
    ]
)

print(completion.choices[0].message.content)

################################################################


################################################################
###  OpenAI数据向量化
def openai_embedding(text: str, model: str=None):
    # 获取环境变量 OPENAI_API_KEY
    api_key=os.environ['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key)

    # embedding model：'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
    if model == None:
        model="text-embedding-3-small"

    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response

response = openai_embedding(text='这行文字将会被向量化')

print(f'向量化后的数据={response.data[0].embedding[:10]}')


