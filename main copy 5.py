#################################################################
# Definindo a resposta do modelo com um objeto JSON             #
#################################################################
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel  # Para validação de dados estruturados
from dotenv import load_dotenv
from langchain.globals import set_debug
import os


# Ativa o modo debug do LangChain para ver detalhes da execução
set_debug(True)


# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI
api_key = os.getenv("OPENAI_API_KEY")

# Define a estrutura dos dados de saída usando Pydantic
# Isso garante que a resposta do modelo siga um formato específico
class Destino(BaseModel):
    cidade: str = Field('A cidade recomendada para visitar')  # Campo para o nome da cidade
    motivo: str = Field('motivo pelo qual é interessante visitar essa cidade')  # Campo para justificativa
    
# Cria um parser JSON que espera receber dados no formato da classe Destino
# Ele converte a resposta em texto do modelo para um objeto JSON estruturado
parseador = JsonOutputParser(pydantic_object=Destino)


# Cria o template de prompt com duas variáveis:
# - {interesse}: será preenchida dinamicamente pelo usuário
# - {formato_de_destino}: será preenchida automaticamente com instruções de formatação JSON
prompt_cidade = PromptTemplate(
    template='''
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_destino}
    ''',
    input_variables=['interesse'],  # Variável que o usuário deve fornecer
    partial_variables={'formato_de_destino': parseador.get_format_instructions()}  
    # get_format_instructions() gera instruções automáticas para o modelo retornar JSON no formato correto
)


# Inicializa o modelo de linguagem da OpenAI
model = ChatOpenAI(
    model='gpt-3.5-turbo',  # Modelo GPT-3.5 Turbo
    temperature=0.5,  # Temperatura média (equilíbrio entre criatividade e consistência)
    api_key=api_key  
)


# Cria a cadeia de processamento (pipeline):
# 1. prompt_cidade: formata o prompt com as variáveis
# 2. model: processa o prompt e gera uma resposta
# 3. parseador: converte a resposta em um objeto JSON estruturado
cadeia = prompt_cidade | model | parseador


# Executa a cadeia fornecendo o interesse 'praias'
response = cadeia.invoke(
    {
        'interesse': 'praias'  # Substitui {interesse} no template
    }
)


# Exibe a resposta estruturada em formato de dicionário Python
# Exemplo esperado: {'cidade': 'Florianópolis', 'motivo': 'Conhecida por suas praias...'}
print(response)