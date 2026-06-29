from flask import Flask, jsonify, request
import numpy as np
import google.generativeai as generativeai
from google import genai
from google.genai import types
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
import os
from rag import buscar_contexto, melhorarResposta

load_dotenv()
app = Flask(__name__)
CORS(app)  # Initialize CORS for the entire application
modelo = 'gemini-3-flash-preview'
chave_secreta = os.getenv('GEMINI_API_KEY')
generativeai.configure(api_key=chave_secreta)


@app.route("/")
def home():
    consulta = "Tem informações da Estação 7 - Loja Nova Resende?"
    resposta = buscar_contexto(consulta)
    print(resposta['documents'][0],"\n\n")
    prompt = f"Consulta: {consulta} Resposta: {resposta['documents'][0]}"
    response = melhorarResposta(prompt)
    return response


@app.route("/api", methods=["POST"])
def results():
    # Verifique a chave de autorização
    auth_key = request.headers.get("Authorization")
    if auth_key != chave_secreta:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True)
    consulta = data["consulta"]
    resultado = buscar_contexto(consulta)
    prompt = f"Consulta: {consulta} Resposta: {resultado['documents'][0]}"
    response = melhorarResposta(prompt)
    return jsonify({"mensagem":  response})


