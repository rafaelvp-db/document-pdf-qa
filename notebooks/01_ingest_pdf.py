# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Ingesting Text from PDF Files
# MAGIC
# MAGIC To ingest text from PDF, there are two possible approaches:
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC 1. Extracting the text from the PDF metadata (easiest case)
# MAGIC   - Caveat: we miss contextual info and structure, such as tables etc.
# MAGIC 2. OCR on top of PDF Files (in case PDFs are image based)
# MAGIC   - Caveat: worse performance

# COMMAND ----------

# DBTITLE 1,Copying Sample PDF files from Repo to DBFS
!mkdir /dbfs/tmp/pdf
!cp -r /Workspace/Repos/rafael.pierre@databricks.com/document-pdf-qa/sample_docs/*.pdf /dbfs/tmp/pdf 

# COMMAND ----------

# DBTITLE 1,1. Extracting text from PDF Metadata
!pip install PyPDF2

# COMMAND ----------

# For demo purposes, we do it first on a single node fashion.
# Code below can be wrapped in a Pandas UDF for running in a distributed fashion.

from PyPDF2 import PdfReader
import glob

pdf_path = "/dbfs/tmp/pdf/*.pdf"

# creating a pdf reader object

def read_pdf(path: str) -> str:
  """Gets the path to a PDF file as an input and returns the text from it."""

  reader = PdfReader(path)
  pdf_pages = []
  
  # printing number of pages in pdf file
  print(f"Number of pages: {len(reader.pages)}")
    
  for i, page in enumerate(reader.pages):
    print(f"Parsing page {i} from {path}:")
    
    # extracting text from page
    text = page.extract_text()
    pdf_content = {"path": path}
    pdf_content["page_number"] = i
    pdf_content["text"] = text
    pdf_pages.append(pdf_content)

  return pdf_pages

# COMMAND ----------

pdf_pages = []

for pdf_file in glob.glob(pdf_path):
  pdf_dict = read_pdf(pdf_file)
  pdf_pages.append(pdf_dict)

# COMMAND ----------

# DBTITLE 1,Persisting parsed text to Delta
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from functools import reduce

schema = StructType([
  StructField("path", StringType(), False),
  StructField("page_number", IntegerType(), False),
  StructField("text", StringType(), False)
])

df_array = [spark.createDataFrame(page) for page in pdf_pages]

df_merge = reduce(lambda x,y:x.union(y), df_array)

# COMMAND ----------

df_merge.count()

# COMMAND ----------

spark.sql("create schema if not exists pdf")
df_merge.write.saveAsTable("pdf.parsed", mode = "overwrite")

# COMMAND ----------

# DBTITLE 1,2. OCR on top of PDF Files (TODO)


# COMMAND ----------


