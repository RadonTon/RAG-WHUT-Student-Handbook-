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





**3. 向量检索与 TF-IDF 重排**

把用户的提问也转化为 512 维向量，由 Chroma 在本地找出距离最近的 12 个候选片段。程序再用 TF-IDF 比较问题与候选原文的字词匹配程度，重排后把前 4 段连同页码发给 DeepSeek。

`RETRIEVAL_CANDIDATES` 默认是 12，控制本地候选数量。`RAG_CONTEXT_SIZE` 默认是 4，控制最终发给 DeepSeek 的片段数量。同一页可以包含多个片段。



**4. 高维空间降维可视化(作图)**

```python
def plot_vector_space(chunks, embeddings, sample_query, output_img="whut_handbook_pca.png"):

```

用主成分分析算法将肉眼不可见的 512 维向量投影到方差最大的前 2 个主方向（PC1、PC2），生成二维散点图 `whut_handbook_pca.png`

（我也看不懂）



**5. 上下文增强条件生成**

```python
def ask_bare_llm(question):

def ask_rag_llm(vector_db, question, k=RAG_CONTEXT_SIZE, *, reranker=None):
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

在项目目录的终端运行。脚本会先运行 4 道示例题的同题对比，再提示你输入自己的问题。

```bash
python rag_experiment_demo.py
```

想跳过示例题、直接提问时运行

```bash
python rag_experiment_demo.py --interactive
```

自定义模式会显示 RAG 回答，并在下面列出参考资料所在的 PDF 页码。同一页只显示一次。可以连续提问，输入「退出」即可结束。

第一问直接检索。后续提问会参考上一轮完整问题。独立问题用原话检索，清楚的追问会显示补全后的问题，缺少关键条件时会先问一句。回复澄清问题后，程序会继续检索。这个判断会额外请求一次 DeepSeek。


#### 相关内容

**向量表示学习、近邻检索（Top-K）、TF-IDF 重排和主成分分析（PCA）降维可视化**

