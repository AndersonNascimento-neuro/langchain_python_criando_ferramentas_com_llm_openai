#################################################################
# Criando uma sequencia de cadeias com LCEL                     #
#################################################################
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain.globals import set_debug
import os


# Ativa o modo debug para ver detalhes da execução das cadeias
set_debug(True)


# Carrega variáveis de ambiente
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Define a estrutura de dados para a primeira resposta (destino)
class Destino(BaseModel):
    cidade: str = Field('A cidade recomendada para visitar')
    motivo: str = Field('motivo pelo qual é interessante visitar essa cidade')
    
# Define a estrutura de dados para a segunda resposta (restaurantes)
class Restaurantes(BaseModel):
    cidade: str = Field('A cidade recomendada para visitar')
    restaurantes: str = Field('Restaurantes recomendados na cidade')
    
# Cria parsers JSON específicos para cada tipo de resposta
parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)


# CADEIA 1: Sugere uma cidade baseada no interesse do usuário
prompt_cidade = PromptTemplate(
    template='''
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_destino}
    ''',
    input_variables=['interesse'],  # Variável fornecida pelo usuário
    partial_variables={'formato_de_destino': parseador_destino.get_format_instructions()}
)

# CADEIA 2: Sugere restaurantes na cidade retornada pela cadeia 1
# Recebe automaticamente a variável {cidade} da cadeia anterior
prompt_restaurantes = PromptTemplate(
    template='''
    Sugira restaurantes populares entre locais em {cidade}.
    {formato_de_destino}
    ''',

    partial_variables={'formato_de_destino': parseador_restaurantes.get_format_instructions()}
)

# CADEIA 3: Sugere atividades culturais na cidade
# Também recebe {cidade} da cadeia anterior
prompt_cultural = PromptTemplate(
    template='Sugira atividades e locais culturais em {cidade}'
)


# Inicializa o modelo de linguagem usado por todas as cadeias
model = ChatOpenAI(
    model='gpt-3.5-turbo',  
    temperature=0.5, 
    api_key=api_key  
)


# Define as três cadeias individuais:
# CADEIA 1: interesse → cidade + motivo (JSON)
cadeia_1 = prompt_cidade | model | parseador_destino

# CADEIA 2: cidade → cidade + restaurantes (JSON)
cadeia_2 = prompt_restaurantes | model | parseador_restaurantes

# CADEIA 3: cidade → atividades culturais (String)
cadeia_3 = prompt_cultural | model | StrOutputParser()


cadeia = (cadeia_1 | cadeia_2 | cadeia_3)

# Executa a cadeia completa
response = cadeia.invoke(
    {
        'interesse': 'praias'  
    }
)


# Imprime o resultado final
print(response)