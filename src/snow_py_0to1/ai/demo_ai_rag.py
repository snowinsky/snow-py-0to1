import os
import time
import json
import textwrap
import numpy as np
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 加载环境变量
load_dotenv()


class ModelAdapter:
    """通用大模型适配器，支持多种大模型无缝切换"""

    def __init__(self, model_name="openai"):
        """
        初始化模型适配器

        参数:
            model_name: 模型名称
                可选: "openai", "deepseek", "silicon", "qwen"
        """
        self.model_name = model_name
        self.setup_model()

    def setup_model(self):
        """根据模型名称设置对应的模型API参数"""
        # 模型支持的端点配置
        self.model_config = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "model": "gpt-4-turbo",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                "embedding_endpoint": "https://api.openai.com/v1/embeddings",
                "embedding_model": "text-embedding-3-small"
            },
            "deepseek": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "endpoint": "https://api.deepseek.com/chat/completions",
                "model": "deepseek-chat",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                    "Content-Type": "application/json"
                }
            },
            "silicon": {
                "api_key": os.getenv("SILICON_API_KEY"),
                "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
                "model": "silicon-llama3",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('SILICON_API_KEY')}",
                    "Content-Type": "application/json"
                }
            },
            "qwen": {
                "api_key": os.getenv("QWEN_API_KEY"),
                "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                "model": "qwen-turbo",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('QWEN_API_KEY')}",
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "enable"
                }
            }
        }

        # 验证配置
        config = self.model_config.get(self.model_name)
        if not config:
            raise ValueError(f"不支持的模型: {self.model_name}")

        # 确保有API密钥
        if not config.get("api_key"):
            raise ValueError(f"未设置 {self.model_name.upper()}_API_KEY 环境变量")

    def generate(self, messages, max_tokens=1000, temperature=0.7):
        """使用选定模型生成文本"""
        config = self.model_config[self.model_name]

        # 根据模型构建请求体
        payload = {
            "openai": {
                "model": config["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            "deepseek": {
                "model": config["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            "silicon": {
                "model": config["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            "qwen": {
                "model": config["model"],
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "result_format": "message",
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            }
        }[self.model_name]

        try:
            response = requests.post(
                config["endpoint"],
                headers=config["headers"],
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            # 解析不同模型的响应格式
            if self.model_name == "openai" or self.model_name == "deepseek" or self.model_name == "silicon":
                return result["choices"][0]["message"]["content"]
            elif self.model_name == "qwen":
                return result["output"]["text"]
            else:
                return result["choices"][0]["text"]

        except Exception as e:
            raise RuntimeError(f"模型调用失败 ({self.model_name}): {str(e)}")

    def embed(self, text):
        """文本向量化（某些模型可能需要专门的嵌入API）"""
        # 对于支持嵌入API的模型
        if self.model_name == "openai":
            config = self.model_config[self.model_name]
            response = requests.post(
                config["embedding_endpoint"],
                headers=config["headers"],
                json={
                    "input": text,
                    "model": config["embedding_model"]
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        else:
            # 简化版向量生成（实际应用中应使用专用模型）
            np.random.seed(hash(text) % (2 ** 32))
            return np.random.rand(384).tolist()


class DocumentLoader:
    """文档加载器"""

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader
        }

    def load(self, chunk_size=1000, chunk_overlap=200):
        """加载并分割文档"""
        documents = []

        for ext, loader_class in self.loaders.items():
            for file_path in self.data_path.rglob(f"*{ext}"):
                try:
                    print(f"加载文档: {file_path.name}")
                    loader = loader_class(str(file_path))
                    docs = loader.load()

                    # 添加元数据
                    for doc in docs:
                        doc.metadata["source"] = str(file_path)
                        if ext == ".pdf":
                            # 调整页码索引
                            page = doc.metadata.get("page", 0)
                            doc.metadata["page"] = page + 1

                    documents.extend(docs)
                except Exception as e:
                    print(f"加载失败 {file_path}: {e}")

        if not documents:
            return []

        # 分割文档
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)


class RAGSystem:
    """通用RAG系统，支持多模型切换"""

    def __init__(self, data_path=None, llm_model="openai", embedding_model="openai"):
        """
        初始化 RAG 系统

        参数:
            data_path: 知识库目录路径
            llm_model: 语言模型 (openai, deepseek, silicon, qwen)
            embedding_model: 嵌入模型 (目前仅 openai 支持专用嵌入API)
        """
        # 初始化模型
        self.llm_adapter = ModelAdapter(llm_model)
        self.embedding_adapter = ModelAdapter(embedding_model)

        # 向量存储系统
        self.vector_store = None
        self.retriever = None

        # RAG链
        self.rag_chain = None

        # 提示模板
        self.prompt_template = ChatPromptTemplate.from_template(
            """请你作为专业助手，基于以下提供的参考信息回答用户问题。
            如果参考信息不足以回答问题，请说明并建议其他资源。
            回答要简洁准确，使用中文。

            参考信息:
            {context}

            问题: {question}

            回答:"""
        )

        # 初始化知识库
        if data_path:
            self.load_knowledge_base(data_path)

    def load_knowledge_base(self, data_path):
        """加载知识库数据"""
        start_time = time.time()
        print(f"正在加载知识库: {data_path}")

        # 1. 加载文档
        loader = DocumentLoader(data_path)
        chunks = loader.load()

        if not chunks:
            print("警告: 未加载任何文档，将使用基础问答模式")
            return

        print(f"已加载并分割文档为 {len(chunks)} 个文本块")

        # 2. 生成嵌入向量（可能使用不同模型的嵌入API）
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.embedding_adapter.embed(chunk.page_content)
                embeddings.append(embedding)
            except Exception as e:
                print(f"文本块 #{i + 1} 向量化失败: {e}")
                # 使用随机向量替代
                np.random.seed(hash(chunk.page_content) % (2 ** 32))
                embeddings.append(np.random.rand(384).tolist())

        # 3. 创建向量存储
        try:
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(
                    [chunk.page_content for chunk in chunks],
                    embeddings
                )),
                embedding=self.embedding_adapter.embed,  # 用于新查询
                metadatas=[chunk.metadata for chunk in chunks]
            )
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # 返回3个最相关结果
            )
            print("向量存储创建成功")

            # 4. 构建 RAG 链
            self.rag_chain = (
                    {"context": self.retriever, "question": RunnablePassthrough()}
                    | self.prompt_template
                    | (lambda x: self.run_llm(x))  # 使用适配器调用LLM
                    | StrOutputParser()
            )
        except Exception as e:
            print(f"创建向量存储失败: {e}")
            print("退回到基本LLM问答模式")
            self.rag_chain = (
                    self.prompt_template.partial(context="")
                    | (lambda x: self.run_llm(x))
                    | StrOutputParser()
            )

        duration = time.time() - start_time
        print(f"初始化完成 (耗时: {duration:.2f}秒)")

    def run_llm(self, prompt_dict):
        """调用语言模型生成回复"""
        # 构建LLM请求消息
        messages = [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user",
             "content": prompt_dict["messages"] if "messages" in prompt_dict else prompt_dict.to_string()}
        ]

        # 调用适配器生成回复
        return self.llm_adapter.generate(messages, max_tokens=1000)

    def query(self, question):
        """查询 RAG 系统"""
        if not self.rag_chain:
            return {
                "answer": "知识库尚未加载完成",
                "sources": []
            }

        # 1. 获取回答
        start_time = time.time()
        try:
            answer = self.rag_chain.invoke(question)
        except Exception as e:
            answer = f"生成回答时出错: {str(e)}"
        llm_time = time.time() - start_time

        # 2. 获取源文档
        sources = []
        if self.retriever:
            try:
                source_docs = self.retriever.invoke(question)
                sources = self._format_sources(source_docs)
            except Exception:
                pass

        # 3. 返回格式化结果
        return {
            "answer": answer,
            "sources": sources,
            "llm_time": llm_time,
            "llm_model": self.llm_adapter.model_name
        }

    def _format_sources(self, documents):
        """格式化源文档信息"""
        sources = []
        for doc in documents:
            metadata = doc.metadata
            if isinstance(doc, dict):
                content = doc.get("page_content", "")
                metadata = doc.get("metadata", {})
            else:
                content = doc.page_content
                metadata = doc.metadata

            source_path = Path(metadata.get("source", "未知文档"))
            sources.append({
                "file": source_path.name,
                "page": metadata.get("page", "未知页码"),
                "extract": textwrap.shorten(content, width=200, placeholder="...")
            })
        return sources


def create_sample_data(path):
    """创建示例数据"""
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    samples = [
        ("人工智能基础.md", """
        ## 人工智能概述

        人工智能(Artificial Intelligence, AI)是计算机科学的一个分支，
        旨在创造能够执行通常需要人类智能的任务的系统。

        ## 核心技术
        - 机器学习
        - 深度学习
        - 自然语言处理
        """),

        ("大语言模型.pdf", """
        大型语言模型(LLM)是基于Transformer架构的深度学习模型。
        这些模型在大量文本数据上训练，可以生成类人文本、翻译语言、回答问题和总结文档。

        ### 知名模型
        - GPT系列 (OpenAI)
        - LLaMA (Meta)
        - Gemini (Google)
        - Qwen (阿里)
        """)
    ]

    # 创建示例文件（模拟PDF创建）
    for filename, content in samples:
        file_path = path / filename

        # 对PDF文件使用特殊处理
        if filename.endswith(".pdf"):
            # 这里实际应该创建真实PDF，但为简化使用文本文件替代
            file_path = path / (filename.replace(".pdf", ".txt"))
            content = "[PDF内容模拟]\n" + content

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())

    print(f"已创建示例知识库: {path}")


def print_banner():
    """打印系统横幅"""
    print("\n" + "=" * 70)
    print(textwrap.dedent("""
      ____   __    __    ___   _        ______   _____    _____ 
     / __ \ / /   / /   /   | | |      / ____/  / ___/   / ___/ 
    / /_/ // /   / /   / /| | | |     / / __   \__ \    \__ \  
   / _, _// /___/ /___/ ___ | | |___ / /_/ /  ___/ /   ___/ /  
  /_/ |_|/_____/_____/_/  |_| |_____(_)____/  /____/   /____/   
    """))
    print("=" * 70)
    print("通用RAG系统 - 支持多种大模型无缝切换")
    print("支持的模型: OpenAI, DeepSeek, 硅基流动, 通义千问")
    print("=" * 70)


if __name__ == "__main__":
    # 创建知识库目录
    knowledge_path = Path("knowledge_base")
    knowledge_path.mkdir(exist_ok=True)

    # 创建示例数据
    create_sample_data(knowledge_path)

    # 打印系统横幅
    print_banner()

    # 模型选择菜单
    available_models = ["openai", "deepseek", "silicon", "qwen"]
    print("\n请选择语言模型:")
    for i, model in enumerate(available_models):
        print(f"{i + 1}. {model.capitalize()}")

    # 用户选择模型
    model_choice = int(input("\n请输入模型编号 (1-4): ")) - 1
    selected_model = available_models[model_choice] if 0 <= model_choice < len(available_models) else "openai"

    # 初始化RAG系统
    rag = RAGSystem(
        data_path=knowledge_path,
        llm_model=selected_model
    )

    print(f"\n已选择 {selected_model.capitalize()} 模型，系统就绪！")

    # 交互式问答
    while True:
        print("\n" + "=" * 70)
        question = input("请输入问题（输入 'exit' 退出）:\n> ").strip()

        if question.lower() == 'exit':
            break
        if not question:
            continue

        # 查询并显示结果
        response = rag.query(question)

        # 打印结果
        print("\n" + "-" * 70)
        print(f"💡 [{response['llm_model'].upper()}模型回答]:")
        print(textwrap.fill(response["answer"], width=70))
        print(f"\n⏱️ 模型响应时间: {response['llm_time']:.2f}秒")

        # 显示源文档
        if response["sources"]:
            print("\n🔍 参考来源:")
            for i, source in enumerate(response["sources"]):
                print(f"{i + 1}. [{source['file']}] (页: {source['page']})")
                print(f"   {source['extract']}")
        else:
            print("\n⚠️ 本次回答未检索知识库内容")
        print("-" * 70)

    print("\n感谢使用通用RAG系统，再见！")