'''
* RecursiveCharacterTextSplitter 递归字符文本分割
RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
    这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
RecursiveCharacterTextSplitter需要关注的是4个参数：

* separators - 分隔符字符串数组
* chunk_size - 每个文档的字符数量限制
* chunk_overlap - 两份文档重叠区域的长度
* length_function - 长度计算函数
'''
#导入文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
# 知识库中单段文本长度
CHUNK_SIZE = 500

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50
# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)

loader = PyMuPDFLoader("/Users/snow/Documents/简历/纪桂松-cn-java_架构设计开发专家.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()


if __name__ == '__main__':
    for pdf_page in pdf_pages:
        text_splitter.split_text(pdf_page.page_content[0:1000])
        split_docs = text_splitter.split_documents(pdf_pages)
        print(f"切分后的文件数量：{len(split_docs)}")
        print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
        for docSlice in split_docs:
            print('$$$$$$$$$$$$$$$$$$$$$$$$')
            print(docSlice.page_content)


def split_pdf():
    return text_splitter.split_documents(pdf_pages)
