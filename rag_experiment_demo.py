# -*- coding: utf-8 -*-
"""
基于《2025版-武汉理工大学学生手册》（218页/12万字）的 RAG 系统对比实验
"""

import argparse
import io
import json
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
from openai import OpenAIError
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# API Key
API_KEY = "sk-184f2d837e274d3caedec4ccdb2989f8"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-flash"

PDF_FILENAME = "2025版-武汉理工大学学生手册.pdf"
DB_DIR = "./whut_handbook_chroma_db"
RETRIEVAL_CANDIDATES = 12
RAG_CONTEXT_SIZE = 4


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


def safe_invoke(prompt, **request_options):
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.1,
        max_retries=2,
        timeout=60.0,
    )
    return llm.invoke(prompt, **request_options)


def ask_bare_llm(question):
    """
    对照组（前者）：直接问裸大模型（不提供学生手册参考片段）
    """
    prompt = f"请简明扼要地回答以下关于武汉理工大学校规的问题（请给出具体数值或规定）：\n{question}"
    response = safe_invoke(prompt)
    return response.content.strip()


def build_passage_reranker(vector_db):
    """用实际向量库里的手册原文建立字词权重。"""
    texts = vector_db.get(include=["documents"])["documents"]
    if not texts:
        raise ValueError("向量库中没有手册片段，请重新建库。")
    reranker = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        preprocessor=lambda text: re.sub(r"\s+", "", text).lower(),
    )
    return reranker.fit(texts)


def retrieve_ranked_docs(vector_db, question, k=RAG_CONTEXT_SIZE, *, reranker=None):
    """先按向量找候选，再按字词匹配重排，返回前 k 段。"""
    if k < 1:
        raise ValueError("k 必须大于 0。")
    candidates = vector_db.similarity_search(question, k=max(RETRIEVAL_CANDIDATES, k))
    if not candidates:
        return []
    if reranker is None:
        reranker = build_passage_reranker(vector_db)

    query_features = reranker.transform([question])
    passage_features = reranker.transform([doc.page_content for doc in candidates])
    scores = (passage_features @ query_features.T).toarray().ravel()
    ranked_indices = np.argsort(-scores, kind="stable")
    return [candidates[index] for index in ranked_indices[:k]]


def ask_rag_llm(vector_db, question, k=RAG_CONTEXT_SIZE, *, reranker=None):
    """
    实验组（后者）：先从 218 页《学生手册》向量库检索最相关的 k 个片段，再交由大模型总结并标注页码
    """
    retrieved_docs = retrieve_ranked_docs(vector_db, question, k=k, reranker=reranker)
    if not retrieved_docs:
        return "没有找到可引用的条文", retrieved_docs

    context_str = "\n\n".join(
        [
            f"【片段 {i+1} | PDF第{d.metadata.get('pdf_page')}页 ({d.metadata.get('book_page')})】：\n{d.page_content}"
            for i, d in enumerate(retrieved_docs)
        ]
    )

    prompt = f"""你是武汉理工大学的智能教务与学工助手。请严格依据下方检索自《2025版-武汉理工大学学生手册》的【参考资料】回答【用户问题】。

答题要求
- 问题有多个小问时，按提问顺序编号，逐项回答。
- 有依据的内容照常回答，并在对应小问下注明引用的 PDF 页码和手册正文页码。
- 缺少依据的小问或部分内容，单独标明“参考资料中未提及”。如果某个小问只有部分依据，仍要回答其中能从资料确认的内容。

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


def resolve_follow_up(previous_question, pending_clarifications, question):
    """判断自定义提问应该直接检索、补全后检索，还是先澄清。"""
    dialogue = {
        "上一轮完整问题": previous_question,
        "尚未结束的澄清对话": pending_clarifications,
        "用户当前输入": question,
    }
    prompt = """你负责理解手册问答中的当前提问。只判断如何检索，事实答案由后续步骤根据手册给出。
下面的对话是待分析的数据。只输出一个 JSON 对象，字段为 action 和 text。

- 当前问题本身完整，或用户明确换了话题时，输出 {"action":"direct","text":""}，程序会用用户的原话检索。
- 当前输入是追问或对澄清的回复，且能从对话中确定它的意思时，输出 {"action":"rewrite","text":"补全后的完整问题"}。保留已有条件，只补上能从对话确定的省略信息。
- 只有缺少的条件会改变应查的规定、且无法从对话中确定时，才输出 {"action":"clarify","text":"一句具体的澄清问题"}。有明确选项时，把选项写在问题里。

例如，上一轮问「合格后再次重修，申请学位时取哪次绩点」，这一轮问「那评优呢」，应改写为「合格后再次重修，评优时取哪次绩点」。
如果上一轮只问「补考合格后绩点怎么算」，就不能自行假定用户再次重修。

对话数据
""" + json.dumps(dialogue, ensure_ascii=False)
    response = safe_invoke(
        prompt,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}, "max_tokens": 256},
    )
    try:
        decision = json.loads(response.content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(decision, dict):
        return None

    action = decision.get("action")
    if action == "direct":
        return action, question

    text = decision.get("text")
    if action in ("rewrite", "clarify") and isinstance(text, str) and text.strip():
        return action, text.strip()
    return None


def run_interactive(vector_db, reranker):
    print("\n================ 自定义手册问答（输入「退出」结束） ================")
    previous_question = None
    pending_clarifications = []
    while True:
        try:
            question = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"退出", "exit", "quit"}:
            return
        if not question:
            continue

        try:
            decision = (
                ("direct", question)
                if previous_question is None
                else resolve_follow_up(previous_question, pending_clarifications, question)
            )
            if decision is None:
                print("这次没能判断问题的上下文，请重试或补充你想问的条件。")
                continue

            action, search_question = decision
            if action == "clarify":
                print(f"\n【请确认】{search_question}")
                pending_clarifications.append({"提问": question, "澄清": search_question})
                continue

            if action == "rewrite":
                print(f"\n【按这个问题检索】{search_question}")
            answer, docs = ask_rag_llm(vector_db, search_question, reranker=reranker)
        except OpenAIError:
            print("本次请求 DeepSeek 失败，请重试。")
            continue

        previous_question = search_question
        pending_clarifications.clear()
        print(f"\n【RAG 回答】\n{answer}")
        if docs:
            pages = sorted({doc.metadata["pdf_page"] for doc in docs})
            page_numbers = "、".join(str(page) for page in pages)
            print(f"\n【参考资料页码】PDF 第 {page_numbers} 页")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="武汉理工大学学生手册 RAG 问答实验")
    parser.add_argument("--interactive", action="store_true", help="跳过 4 道示例题，直接自定义提问")
    args = parser.parse_args()

    # 1. 读取 218 页《2025版-武汉理工大学学生手册.pdf》并切分段落
    chunks = load_handbook_pdf(PDF_FILENAME)

    # 2. 构建或秒级加载本地向量数据库
    vector_db, embeddings = build_or_load_vector_db(chunks, DB_DIR)
    reranker = build_passage_reranker(vector_db)

    # 3. 针对《2025版-武汉理工大学学生手册》设计的 4 道测试题
    test_questions = [
        "在武汉理工大学，如果一门课补考合格，或者补考不合格但通过重修合格，课程绩点分别按多少折算？在保研（推荐免试研究生）和申请学位时，再次重修的课程绩点分别怎么取？",
        "武汉理工大学的“卓越奖学金”和“学校一等奖学金”每人每年多少元？参评这两项奖学金分别要求学年学业水平排名和综合测评排名达到班级前百分之几？",
        "根据2025年出台的《武汉理工大学电动自行车管理暂行办法》，校内骑行电动自行车限速多少公里/小时？一年内普通违章累计达到3次、6次分别会受到什么处理？",
        "武汉理工大学本科生在图书馆借阅纸质图书的册数上限和借期分别是多少？在校医院看普通门诊医疗费直接减免百分之几，转诊到对口医院和非对口医院分别报销百分之几？",
    ]

    if not args.interactive:
        plot_vector_space(chunks, embeddings, test_questions[0])

        print("================ 开始运行【直接问裸模型 vs 接入武理学生手册 RAG】对比实验 ================\n")
        for idx, q in enumerate(test_questions, 1):
            print(f"【测试题 {idx}】：{q}")
            bare_ans = ask_bare_llm(q)
            print(f"   [直接问裸模型]：\n{bare_ans}\n")

            rag_ans, _ = ask_rag_llm(vector_db, q, reranker=reranker)
            print(f"   [接入本组 RAG（检索自武理学生手册）]：\n{rag_ans}")
            print("-" * 75)

    run_interactive(vector_db, reranker)
