#################################################################
# Simulando uma interação de chat sem memória                   #
#################################################################

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Inicializa o modelo de chat da OpenAI
modelo = ChatOpenAI(
    model='gpt-3.5-turbo',  # Modelo GPT-3.5 Turbo
    temperature=0.5,  # Temperatura média para equilíbrio entre criatividade e consistência
    api_key=api_key  # Chave da API
)

# Lista de perguntas a serem feitas ao modelo
# A segunda pergunta faz referência à primeira, mas o modelo NÃO tem memória
lista_perguntas = [
    'Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?',
    'Qual a melhor época do ano para ir?'  # ⚠️ Esta pergunta depende do contexto anterior
]

# Itera sobre cada pergunta da lista
for pergunta in lista_perguntas:
    # Envia a pergunta ao modelo (sem contexto das perguntas anteriores)
    resposta = modelo.invoke(pergunta)
    
    # Exibe a pergunta do usuário
    print(f'Usuário: {pergunta}')
    
    # Exibe a resposta do modelo (resposta.content contém o texto)
    print(f'IA: {resposta.content}\n')
    
# ⚠️ PROBLEMA: O modelo NÃO tem memória entre as iterações
# Na segunda pergunta, o modelo não sabe qual cidade foi sugerida na primeira
# Ele responderá de forma genérica ou pedirá mais informações
