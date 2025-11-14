######################################################
# Orquestrando assistentes sem LangGraph             #
######################################################

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import Literal, TypedDict
import os


# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Inicializa o modelo de chat da OpenAI
modelo = ChatOpenAI(
    model='gpt-4o-mini',  # Modelo GPT-4o-mini
    temperature=0.5,  # Temperatura média
    api_key=api_key
)

# Prompt para o especialista em PRAIAS
# Define a personalidade como "Sra Praia"
prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ('system', 'Apresente-se como Sra Praia. Você é uma especialista em viagens com destinos para praia.'),
        ('human', '{query}')  # Pergunta do usuário
    ]
)

# Prompt para o especialista em MONTANHAS
# Define a personalidade como "Sr Montanha"
prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ('system', 'Apresente-se como Sr Montanha. Você é um especialista em viagens com destinos para montanhas.'),
        ('human', '{query}')  # Pergunta do usuário
    ]
)

# Cadeia para destinos de PRAIA
# prompt → modelo → parser (string)
cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()

# Cadeia para destinos de MONTANHA
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

# Define a estrutura de dados para a rota
# TypedDict especifica que 'destino' só pode ser 'praia' ou 'montanha'
class Rota(TypedDict):
    destino: Literal['praia', 'montanha']  # Apenas esses dois valores são válidos
    
# Prompt do ROTEADOR
# Responsável por classificar a pergunta do usuário
# Determina se a pergunta é sobre praia ou montanha
prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ('system', 'Responda apenas com "praia" ou "montanha"'),  # Instrução restritiva
        ('human', '{query}')  # Pergunta a ser classificada
    ]
)

# Cria o roteador com saída estruturada
# with_structured_output(Rota) força o modelo a retornar um objeto no formato da classe Rota
# Isso garante que a resposta será sempre {'destino': 'praia'} ou {'destino': 'montanha'}
roteador = prompt_roteador | modelo.with_structured_output(Rota)

# Função principal que coordena todo o fluxo
def responda(pergunta: str):
    # PASSO 1: Usa o roteador para classificar a pergunta
    # Retorna um dict: {'destino': 'praia'} ou {'destino': 'montanha'}
    rota = roteador.invoke({'query': pergunta})['destino']
    
    # PASSO 2: Baseado na classificação, direciona para o especialista correto
    if rota == 'praia':
        # Se a pergunta é sobre praia, usa a Sra Praia
        return cadeia_praia.invoke({'query': pergunta})
    
    # Se a pergunta é sobre montanha, usa o Sr Montanha
    return cadeia_montanha.invoke({'query': pergunta})

# Testa a função com uma pergunta sobre surf
# "surfar" e "lugar quente" indica praia → será direcionado para Sra Praia
print(responda('Quero surfar em um lugar quente.'))

# Teste alternativo comentado
# Esta pergunta também seria direcionada para a Sra Praia
# print(responda('Quero passear por praias belas no Brasil.'))

'''
## **Como funciona o fluxo de roteamento:**
PERGUNTA: "Quero surfar em um lugar quente."
    ↓
ROTEADOR: Classifica a pergunta
    ├─ Analisa palavras-chave: "surfar", "lugar quente"
    ├─ Determina: isto é sobre PRAIA
    └─ Retorna: {'destino': 'praia'}
    ↓
DECISÃO: rota == 'praia'
    ↓
CADEIA_PRAIA: Sra Praia responde
    ├─ System: "Apresente-se como Sra Praia..."
    ├─ Human: "Quero surfar em um lugar quente."
    └─ Output: "Olá! Sou a Sra Praia... Sugiro Fernando de Noronha..."
'''
