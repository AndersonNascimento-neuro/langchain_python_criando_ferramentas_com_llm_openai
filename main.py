from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os


# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Parâmetros da viagem
numero_dias = 7
numero_criancas = 2
atividade = 'praia'

# Define template reutilizável com placeholders entre chaves
prompt_model = PromptTemplate(
    template='''
    Crie um roteiro de viagem de {dias} dias,
    para uma família com {numero_criancas} crianças, 
    que gostam de {atividade}
    '''
)

# Substitui placeholders pelos valores reais
prompt = prompt_model.format(
    dias=numero_dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)

# Inicializa modelo LangChain (temperature=0.5 equilibra criatividade e consistência)
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0.5,
    api_key=api_key
)

# Envia prompt formatado e exibe resposta
response = model.invoke(prompt)
print(response.content)