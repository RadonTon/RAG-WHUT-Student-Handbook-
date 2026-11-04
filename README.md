### 检索增强生成（RAG）



##### 什么是检索增强生成（RAG)？

简单来说，比如当你想询问大模型关于武汉理工大学学生手册的有关内容时，你大概率得不到准确的答案

你可以在问问题的时候把整个学生手册的pdf一起上传，这得花不少token

RAG作为解决方案，会将整个pdf预处理，在你提问时，本地轻量小模型会从pdf选取和问题最贴近的k个片段一起上传给大模型。



##### 脚本在做什么？

**1. 原生 PDF 解析与切分**

~~~python
 chunks = load_handbook_pdf(PDF_FILENAME)
    
 def load_handbook_pdf(pdf_path=PDF_FILENAME):
   
~~~

自动清理 PDF 尾部破损水印，在 2 秒内提取全部 218 页原生汉字编码（12.1 万字），按每段 `450` 字（重叠 `60` 字）切分为约 400 个带页码标签的小段落



**2. 文本向量化与建库**

```python
def build_or_load_vector_db(chunks, persist_dir=DB_DIR):
```

调用智源开源轻量中文向量模型，从 512 个语义维度给每个段落打分，转化为 **512 维稠密数学向量**，存入本地 `./whut_handbook_chroma_db` 文件夹





**3. 语义相似度近邻检索**

```python
def ask_rag_llm(vector_db, question, k=4):
    
retrieved_docs = vector_db.similarity_search(question, k=k)
```

把用户的提问也转化为 512 维向量，在本地计算问题向量与全书 400 多个段落向量的**夹角余弦值**，挑出语义最接近的前 $K=4$ 个段落



**4. 高维空间降维可视化(作图)**

```python
def plot_vector_space(chunks, embeddings, sample_query, output_img="whut_handbook_pca.png"):

```

用主成分分析算法将肉眼不可见的 512 维向量投影到方差最大的前 2 个主方向（PC1、PC2），生成二维散点图 `whut_handbook_pca.png`

（我也看不懂）



**5. 上下文增强条件生成**

```python
def ask_bare_llm(question):

def ask_rag_llm(vector_db, question, k=4):
```

将同一道题分别以“纯问题（Baseline）”和“4 段手册原文 + 问题（RAG）”发给云端大模型，输出同题对比结果



### 首次运行脚本

**1.在终端运行带清华镜像加速的安装命令**

```python
pip install langchain langchain-community langchain-openai chromadb pypdf sentence-transformers scikit-learn matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

```



**2.填写api key **

```python
API_KEY = "sk-184f2d837e274d3caedec4ccdb2989f8"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-flash"
```



**3.运行**

在终端运行

```python
python rag_experiment_demo.py
```



#### 相关内容

**向量表示学习、余弦相似度度量、**$K$**-近邻检索（Top-**$K$**）以及主成分分析（PCA）降维可视化**

