import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import google.generativeai as generativeai
from google import genai
from google.genai import types


load_dotenv()

chave_secreta = os.environ.get('GEMINI_API_KEY', '')
if not chave_secreta:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it with your Gemini API key.")

generativeai.configure(api_key=chave_secreta)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="ifsuldeminas"
)

csv_url = 'https://docs.google.com/spreadsheets/d/11QU1ibjUAlNKLwLWF1s-kSpRH2UBOiVbLyl1pJIyeSk/export?format=csv&id=11QU1ibjUAlNKLwLWF1s-kSpRH2UBOiVbLyl1pJIyeSk'
df = pd.read_csv(csv_url)
print(df.head())

import google.generativeai as generativeai
import numpy as np

# função  para gerar os embeddings 
def gerarEmbeddings(title, text):
  model = 'models/gemini-embedding-001'
  result = generativeai.embed_content(model=model,
                                content=text,
                                task_type="retrieval_document",
                                title=title)
  return result['embedding']


#aplicando a função para gerar os embeddings e inserir no banco
for index, row in df.iterrows():
    embedding = gerarEmbeddings(
        row["Titulo"],
        row["Conteúdo"]
    )
    collection.add(
        ids=[str(index)],
        documents=[row["Conteúdo"]],
        metadatas=[{
            "titulo": row["Titulo"]
        }],
        embeddings=[embedding]
    )
print("Embeddings salvos no ChromaDB!")