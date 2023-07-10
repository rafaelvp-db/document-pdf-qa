# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Creating embeddings from parsed text
# MAGIC
# MAGIC </br>
# MAGIC
# MAGIC * Here we convert the text parsed from PDFs into embeddings
# MAGIC * For this purpose, we will use [ChromaDB](https://docs.trychroma.com/)
# MAGIC * **ChromaDB** is a lightweight Vector DB. It is the equivalent of SQLite in the relational world - so don't use it in production 😀
# MAGIC   * By default, Chroma uses `all-MiniLM-L6-v2` embeddings from [Sentence Transformers](https://www.sbert.net/examples/applications/computing-embeddings/README.html).
# MAGIC   * Max sequence length for `all-MiniLM-L6-v2` is 128, so we will need to break our text into chunks - at least for a first iteration

# COMMAND ----------

# MAGIC %pip install -U -q chromadb typing-extensions

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Breaking PDF text pages into smaller chunks
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC * There are different approaches when it comes to breaking text down into chunks for LLM and Semantic Search applications, as described [here](https://www.pinecone.io/learn/chunking-strategies/)
# MAGIC * We will start with a basic/naive approach - breaking text into sentences
# MAGIC * Let's start with an example for one of the documents

# COMMAND ----------

text = (
  spark
    .sql(
      "select text from pdf.parsed where path like '%nasdaq_composite%'"
    ).collect()
)

nasdaq_page_one = text[0].__getattr__("text")
print(nasdaq_page_one)

# COMMAND ----------

from textwrap import wrap

n = 128

nasdaq_chunks = wrap(
    nasdaq_page_one,
    n,
    drop_whitespace=False,
    break_on_hyphens=False
)

print(nasdaq_chunks)

# COMMAND ----------

# DBTITLE 1,Converting to Embeddings and Indexing on ChromaDB
"""
By default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings. This embedding model can create sentence and document embeddings that can be used for a wide variety of tasks. This embedding function runs locally on your machine, and may require you download the model files (this will happen automatically).
"""

import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="pdf_files")

index_arr = [f"nasdaq_page_1_chunk_{c}" for c in range(len(nasdaq_chunks))]

collection.add(
    documents=nasdaq_chunks,
    metadatas=[{"id": index} for index in index_arr],
    ids=index_arr
)

# COMMAND ----------

# DBTITLE 1,Running queries
results = collection.query(
    query_texts=["What was the Nasdaq Composite Monthly performance for December 2018?"],
    n_results=5
)

results["documents"]

# COMMAND ----------

#TODO: use LLMs for formatting the right part of the answer - examples:

# - https://huggingface.co/impira/layoutlm-document-qa

# COMMAND ----------


