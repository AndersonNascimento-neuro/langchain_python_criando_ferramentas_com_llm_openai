from openai import OpenAI
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Parâmetros da viagem definidos manualmente
numero_dias = 7  # Duração da viagem
numero_criancas = 2  # Quantidade de crianças na família
atividade = 'música'  # Interesse principal da família

# Template do prompt com placeholders entre chaves {}
prompt = f'Crie um roteiro de viagem de {numero_dias} dias, para uma família com {numero_criancas} crianças, que gosta de {atividade}.'

# Inicializa o cliente da API OpenAI
client = OpenAI(api_key=api_key)

# Faz a chamada à API da OpenAI
# A comunicação usa um formato de mensagens com roles (papéis)
response = client.chat.completions.create(
    model='gpt-3.5-turbo',  # Modelo utilizado
    messages=[
        {
            'role': 'system',  # Mensagem do sistema: define o comportamento do assistente
            'content': 'Você é um assistente de roteiro de viagens.'
        },
        {
            'role': 'user',  # Mensagem do usuário: o prompt/pergunta real
            'content': prompt  # Envia o prompt (mas com as variáveis NÃO substituídas)
        }
    ]
)

# Extrai o conteúdo da resposta
# choices[0] = primeira (e única) resposta gerada
# message.content = texto da resposta
resposta_em_texto = response.choices[0].message.content
print(resposta_em_texto)