#################################################################
# Criando cadeias com LCEL no LangChain                         #
#################################################################
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# Carrega variáveis de ambiente de um arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI das variáveis de ambiente
api_key = os.getenv("OPENAI_API_KEY")

# Cria um template de prompt para sugestão de cidades
# O template possui uma variável {interesse} que será substituída dinamicamente
prompt_cidade = PromptTemplate(
    template='''
    Sugira uma cidade dado o meu interesse por {interesse}.
    ''',
    input_variables=['interesse']  # Define que 'interesse' é uma variável de entrada
)

# Inicializa o modelo de chat da OpenAI
model = ChatOpenAI(
    model='gpt-3.5-turbo',  # Especifica o modelo GPT-3.5 Turbo
    temperature=0.5,  # Controla a criatividade (0 = determinístico, 1 = mais criativo)
    api_key=api_key  # Passa a chave da API
)

# Cria uma cadeia (chain) usando o operador pipe (|)
# Fluxo: prompt_cidade → model → StrOutputParser()
# 1. O prompt é formatado com os valores fornecidos
# 2. O modelo processa o prompt
# 3. O parser converte a resposta em string
cadeia = prompt_cidade | model | StrOutputParser()

# Executa a cadeia com o interesse 'praias'
response = cadeia.invoke(
    {
        'interesse': 'praias'  # Substitui {interesse} no template por 'praias'
    }
)

# Exibe a resposta do modelo (sugestão de cidade)
print(response)