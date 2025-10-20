from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Parâmetros da viagem
numero_dias = 7
numero_criancas = 2
atividade = 'praia'

prompt = f'Crie um roteiro de viagens, para um período de {numero_dias}, para uma família com {numero_criancas} que busca atividades relacinadas a {atividade}'

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0.5,
    api_key=api_key
)

response = model.invoke(prompt)
print(response.content)