from openai import OpenAI
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Parâmetros da viagem
numero_dias = 7
numero_criancas = 2
atividade = 'música'

# Template do prompt 
prompt = 'Crie um roteiro de viagem de {numero_dias} dias, para uma família com {nuumero_criancas} crianças, que gosta de {atividade}.'

# Inicializa cliente da API OpenAI
client = OpenAI(api_key=api_key)

# Faz requisição à API com contexto de sistema e prompt do usuário
response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            'role': 'system',
            'content': 'Você é um assistente de roteiro de viagens.'
        },
        {
            'role': 'user',
            'content': prompt
        }
    ]
)

# Extrai e exibe o texto da resposta
resposta_em_texto = response.choices[0].message.content
print(resposta_em_texto)