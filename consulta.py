from rag import buscar_contexto, melhorarResposta
consulta = "Tem informações da Estação 7 - Loja Nova Resende?"
resposta = buscar_contexto(consulta)
print(resposta['documents'][0],"\n\n")
prompt = f"Consulta: {consulta} Resposta: {resposta['documents'][0]}"
print(prompt,"\n\n")
print("Melhorando resposta...")
response = melhorarResposta(prompt)
print(response)

