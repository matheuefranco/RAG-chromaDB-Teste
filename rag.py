
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

import google.generativeai as generativeai
import numpy as np

# Função para gerar resposta a partir da consulta
def buscar_contexto(consulta):
    model = 'models/gemini-embedding-001'
    embedding_consulta = generativeai.embed_content(
        model=model,
        content=consulta,
        task_type="retrieval_query"
    )['embedding']
    resultados = collection.query(
        query_embeddings=[embedding_consulta],
        n_results=2
    )
    return resultados



def melhorarResposta(inputText):
    modelo = 'gemini-3-flash-preview'
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = modelo
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=inputText),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
      types.Part.from_text(text="""
      Você é um assistente baseado em RAG (Retrieval-Augmented Generation).
      Utilize exclusivamente o conteúdo recuperado da base de conhecimento para responder à consulta do usuário.
      Considere a consulta e o contexto recuperado, e gere uma resposta clara, objetiva e coerente, reescrevendo as informações de forma natural sem copiar literalmente o texto original.
      Não invente informações que não estejam presentes no contexto fornecido e não apresente múltiplas opções de resposta.
      """),
      ],
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text;
#fim da função para melhorar a resposta

