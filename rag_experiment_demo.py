# -*- coding: utf-8 -*-
"""
基于《2025版-武汉理工大学学生手册》（218页/12万字）的 RAG 系统对比实验
"""

import io
import os
import re
import time

# 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# API Key
API_KEY = "sk-184f2d837e274d3caedec4ccdb2989f8"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-flash"

PDF_FILENAME = "2025版-武汉理工大学学生手册.pdf"
DB_DIR = "./whut_handbook_chroma_db"


def load_handbook_pdf(pdf_path=PDF_FILENAME):
    """
    第一步：自动修复并读取 218 页《2025版-武汉理工大学学生手册.pdf》原生文字层。
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"未在当前文件夹找到【{pdf_path}】！请将《2025版-武汉理工大学学生手册.pdf》复制到本脚本同一文件夹下。"
        )

    print(f"正在读取并自动修复《{pdf_path}》...")
    with open(pdf_path, "rb") as f:
        raw_bytes = f.read()

    # 自动清理 PDF 尾部 startxref 后粘连的 %WUT-FREE-SHARE-TRACE 水印注释
    cleaned_bytes = re.sub(rb"(\d+)%WUT-FREE-SHARE-TRACE[^\r\n]*", rb"\1", raw_bytes)
    reader = PdfReader(io.BytesIO(cleaned_bytes))

    raw_docs = []
    total_chars = 0
    for idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            total_chars += len(text)
            pdf_page_num = idx + 1
            book_page_str = f"正文第{pdf_page_num - 7}页" if pdf_page_num >= 8 else "前言/目录"
            raw_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "pdf_page": pdf_page_num,
                        "book_page": book_page_str,
                    },
                )
            )

    print(f"成功提取 {len(raw_docs)} 页原生文字，总字数：{total_chars:,} 字！")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60,
        separators=["\n\n", "\n", "。", "；", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"语料切分完成：共生成 {len(chunks)} 个文本片段（Chunks）。\n")
    return chunks


def build_or_load_vector_db(chunks, persist_dir=DB_DIR):
    """
    第二步：构建本地持久化向量数据库。
    """
    print("正在加载本地轻量级中文向量模型 (BAAI/bge-small-zh-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"检测到本地已建好的向量库 {persist_dir}\n")
        vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print(f"首次运行：正在将 {len(chunks)} 个手册片段转化为 512 维数学向量并存入 {persist_dir}")
        vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
        print("向量数据库构建并保存完成！\n")

    return vector_db, embeddings


def safe_invoke(prompt):
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.1,
        max_retries=2,
        timeout=60.0,
    )
    return llm.invoke(prompt)


def ask_bare_llm(question):
    """
    对照组（前者）：直接问裸大模型（不提供学生手册参考片段）
    """
    prompt = f"请简明扼要地回答以下关于武汉理工大学校规的问题（请给出具体数值或规定）：\n{question}"
    response = safe_invoke(prompt)
    return response.content.strip()


def ask_rag_llm(vector_db, question, k=4):
    """
    实验组（后者）：先从 218 页《学生手册》向量库检索最相关的 k 个片段，再交由大模型总结并标注页码
    """
    retrieved_docs = vector_db.similarity_search(question, k=k)
    context_str = "\n\n".join(
        [
            f"【片段 {i+1} | PDF第{d.metadata.get('pdf_page')}页 ({d.metadata.get('book_page')})】：\n{d.page_content}"
            for i, d in enumerate(retrieved_docs)
        ]
    )

    prompt = f"""你是武汉理工大学的智能教务与学工助手。请严格依据下方检索自《2025版-武汉理工大学学生手册》的【参考资料】回答【用户问题】，并在回答末尾准确注明引用的 PDF 页码及手册正文页码。
如果参考资料中没有相关信息，请明确回答“参考资料中未提及”。

【参考资料】：
{context_str}

【用户问题】：
{question}"""

    response = safe_invoke(prompt)
    return response.content.strip(), retrieved_docs


def plot_vector_space(chunks, embeddings, sample_query, output_img="whut_handbook_pca.png"):
    """
    绘制 2D PCA 向量空间散点图
    """
    sample_chunks = chunks[:80] if len(chunks) > 80 else chunks
    texts = [c.page_content for c in sample_chunks]
    doc_vectors = np.array(embeddings.embed_documents(texts))
    query_vector = np.array(embeddings.embed_query(sample_query)).reshape(1, -1)

    all_vectors = np.vstack([doc_vectors, query_vector])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_vectors)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(reduced[:-1, 0], reduced[:-1, 1], c="#3b82f6", s=45, alpha=0.65, label="WHUT Handbook Chunks")
    plt.scatter(reduced[-1, 0], reduced[-1, 1], c="#ef4444", s=200, marker="*", label="User Query Vector")
    plt.annotate("User Query", (reduced[-1, 0] + 0.015, reduced[-1, 1] + 0.015), fontsize=10, fontweight="bold", color="#ef4444")

    plt.title("2D PCA Projection of WHUT Student Handbook Embeddings", fontsize=12)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"已生成《武汉理工大学学生手册》向量空间 PCA 降维图：{output_img}\n")


if __name__ == "__main__":
    # 1. 读取 218 页《2025版-武汉理工大学学生手册.pdf》并切分段落
    chunks = load_handbook_pdf(PDF_FILENAME)

    # 2. 构建或秒级加载本地向量数据库
    vector_db, embeddings = build_or_load_vector_db(chunks, DB_DIR)

    # 3. 针对《2025版-武汉理工大学学生手册》设计的 4 道测试题
    test_questions = [
        "在武汉理工大学，如果一门课补考合格，或者补考不合格但通过重修合格，课程绩点分别按多少折算？在保研（推荐免试研究生）和申请学位时，再次重修的课程绩点分别怎么取？",
        "武汉理工大学的“卓越奖学金”和“学校一等奖学金”每人每年多少元？参评这两项奖学金分别要求学年学业水平排名和综合测评排名达到班级前百分之几？",
        "根据2025年出台的《武汉理工大学电动自行车管理暂行办法》，校内骑行电动自行车限速多少公里/小时？一年内普通违章累计达到3次、6次分别会受到什么处理？",
        "武汉理工大学本科生在图书馆借阅纸质图书的册数上限和借期分别是多少？在校医院看普通门诊医疗费直接减免百分之几，转诊到对口医院和非对口医院分别报销百分之几？",
    ]

    # 生成高维向量降维可视化图
    plot_vector_space(chunks, embeddings, test_questions[0])

    # 4. 运行【直接问裸模型和接入武理学生手册 RAG】同题对比实验

    print("================ 开始运行【直接问裸模型 vs 接入武理学生手册 RAG】对比实验 ================\n")
    for idx, q in enumerate(test_questions, 1):
        print(f"【测试题 {idx}】：{q}")
        bare_ans = ask_bare_llm(q)
        print(f"   [直接问裸模型]：\n{bare_ans}\n")

        rag_ans, docs = ask_rag_llm(vector_db, q, k=4)
        print(f"   [接入本组 RAG（检索自武理学生手册）]：\n{rag_ans}")
        print("-" * 75)