######################################################
# Criando a base de uma solução com LCEL e LangGraph #
######################################################

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Inicializa o modelo de chat da OpenAI
# Usando GPT-4o-mini (versão mais leve e econômica do GPT-4)
modelo = ChatOpenAI(
    model='gpt-4o-mini',  # Modelo GPT-4o-mini - mais rápido e barato que o GPT-4 completo
    temperature=0.5,  # Temperatura média - equilíbrio entre criatividade e consistência
    api_key=api_key  # Chave da API
)

# Define o template de prompt com duas mensagens:
# 1. Mensagem do sistema: define o papel/comportamento do assistente
# 2. Mensagem humana: a pergunta do usuário (variável {query})
prompt_consultor = ChatPromptTemplate.from_messages(
    [
        ('system', 'Você é um consultor de viagens'),  # Contexto do sistema
        ('human', '{query}')  # Placeholder para a pergunta do usuário
    ]
)

# Cria a cadeia de processamento (pipeline):
# 1. prompt_consultor: formata o prompt com a query do usuário
# 2. modelo: processa o prompt e gera a resposta
# 3. StrOutputParser(): converte a resposta em string simples
assistente = prompt_consultor | modelo | StrOutputParser()

# Invoca a cadeia passando a query do usuário
# A query substituirá o placeholder {query} no prompt
response = assistente.invoke(
    {'query': 'Quero férias em praias no Brasil.'}
)

# Exibe a resposta do modelo
print(response)

# 📝 NOTA: Este código NÃO tem memória
# Cada invoke() é uma interação independente, sem contexto de mensagens anteriores
# Para adicionar memória, seria necessário usar RunnableWithMessageHistory
