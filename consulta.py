from rag import buscar_contexto, melhorarResposta
consulta = "Em quais cidades fica o IF?"
resposta = buscar_contexto(consulta)
print(resposta['documents'][0],"\n\n")
prompt = f"Consulta: {consulta} Resposta: {resposta['documents'][0]}"
print(prompt,"\n\n")
print("Melhorando resposta...")
response = melhorarResposta(prompt)
print(response)

