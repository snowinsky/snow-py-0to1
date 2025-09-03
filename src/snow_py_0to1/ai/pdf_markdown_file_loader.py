from langchain_community.document_loaders import PyMuPDFLoader
####################### 加载 pdf 格式的文件 #############################
# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("/Users/snow/Documents/简历/纪桂松-cn-java_架构设计开发专家.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()

print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")

pdf_page = pdf_pages[0]
print(f"每一个元素的类型：{type(pdf_page)}.",
    f"该文档的描述性数据：{pdf_page.metadata}",
    f"查看该文档的内容:\n{pdf_page.page_content}",
    sep="\n------\n")
####################### 加载 markdown 格式的文本文件 #############################
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("/Users/snow/snow_doc/snow_doc/简历/英文简历/jiguisong-en-英文简历.md")
markdown_pages = loader.load()
md_page = markdown_pages[0]
print(f"每一个元素的类型：{type(md_page)}.",
    f"该文档的描述性数据：{md_page.metadata}",
    f"查看该文档的内容:\n{md_page.page_content[0:][:200]}",
    sep="\n------\n")



