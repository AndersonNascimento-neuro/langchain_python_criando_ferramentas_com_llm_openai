######################################################
# Orquestrando assistentes sem LangGraph             #
######################################################

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

import asyncio
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


# Prompt para especialista em PRAIAS
prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ('system', 'Apresente-se como Sra Praia. Você é uma especialista em viagens com destinos para praia.'),
        ('human', '{query}')
    ]
)

# Prompt para especialista em MONTANHAS
prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ('system', 'Apresente-se como Sr Montanha. Você é um especialista em viagens com destinos para montanhas.'),
        ('human', '{query}')  # Pergunta do usuário
    ]
)

# Cadeia para processar consultas sobre PRAIAS
cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()

# Cadeia para processar consultas sobre MONTANHAS
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

# Define a estrutura de dados para classificação de rota
# Só aceita 'praia' ou 'montanha' como valores válidos
class Rota(TypedDict):
    destino: Literal['praia', 'montanha']

# Prompt do roteador - classifica a pergunta do usuário
prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ('system', 'Responda apenas com "praia" ou "montanha"'),
        ('human', '{query}')
    ]
)

# Roteador com saída estruturada garantida
roteador = prompt_roteador | modelo.with_structured_output(Rota)


# Define a estrutura do ESTADO do grafo
# O estado mantém todas as informações que fluem pelo grafo
class Estado(TypedDict):
    query: str  # A pergunta do usuário
    destino: Rota  # A classificação (praia ou montanha)
    resposta: str  # A resposta final do especialista
    

# Nó ROTEADOR - Classifica a pergunta do usuário
# Função assíncrona que determina se a pergunta é sobre praia ou montanha
async def no_roteador(estado: Estado, config=RunnableConfig):
    return {
        'destino': await roteador.ainvoke(
            {'query': estado['query']},  # Passa a pergunta para o roteador
            config  # Configuração de execução
        )
    }
    
# Nó PRAIA - Processa consultas sobre praias
# Função assíncrona que gera resposta da Sra Praia
async def no_praia(estado: Estado, config=RunnableConfig):
    return {
        'resposta': await cadeia_praia.ainvoke(
            {'query': estado['query']},  # Passa a pergunta para a especialista
            config 
        )
    }
    
# Nó MONTANHA - Processa consultas sobre montanhas
# Função assíncrona que gera resposta do Sr Montanha
async def no_montanha(estado: Estado, config=RunnableConfig):
    return {
        'resposta': await cadeia_montanha.ainvoke(
            {'query': estado['query']},  # Passa a pergunta para o especialista
            config 
        )
    }
    
# Função de DECISÃO - Determina qual nó deve ser executado a seguir
# ⚠️ ERRO DE DIGITAÇÃO: "escholher_no" deveria ser "escolher_no"
# Baseado no destino classificado, retorna 'praia' ou 'montanha'
def escholher_no(estado: Estado) -> Literal['praia', 'montanha']:
    return 'praia' if estado['destino']['destino'] == 'praia' else 'montanha'

# Cria o GRAFO de estados com a estrutura definida
grafo = StateGraph(Estado)

# Adiciona os NÓS ao grafo
# Cada nó representa uma etapa do processamento
grafo.add_node('rotear', no_roteador)  # Nó que classifica
grafo.add_node('praia', no_praia)  # Nó especialista em praias
grafo.add_node('montanha', no_montanha)  # Nó especialista em montanhas

# Define as ARESTAS (conexões entre nós)
# START → 'rotear': A execução sempre começa no nó roteador
grafo.add_edge(START, 'rotear')

# 'rotear' → ?: Aresta condicional que decide o próximo nó
# Usa a função escolher_no para determinar se vai para 'praia' ou 'montanha'
grafo.add_conditional_edges('rotear', escholher_no)

# 'praia' → END: Após a Sra Praia responder, finaliza
grafo.add_edge('praia', END)

# 'montanha' → END: Após o Sr Montanha responder, finaliza
grafo.add_edge('montanha', END)

# Compila o grafo em uma aplicação executável
app = grafo.compile()

# Função principal assíncrona que executa o grafo
async def main():
    # Invoca o grafo de forma assíncrona com a pergunta do usuário
    resposta = await app.ainvoke(
        {
            'query': 'Quero escalar montanhas radicais no sul do Brasil'
        }
    )
    # Exibe apenas a resposta final (do especialista)
    print(resposta['resposta'])
    
# Executa a função principal assíncrona
asyncio.run(main())



## **Fluxo de execução do grafo:**
'''
INPUT: "Quero escalar montanhas radicais no sul do Brasil"
    ↓
[START]
    ↓
[NÓ: rotear]
    ├─ Classifica a pergunta
    ├─ Identifica palavras: "escalar", "montanhas"
    └─ Retorna: {'destino': 'montanha'}
    ↓
[DECISÃO CONDICIONAL: escolher_no]
    ├─ Verifica: estado['destino']['destino'] == 'montanha'
    └─ Redireciona para: 'montanha'
    ↓
[NÓ: montanha]
    ├─ Sr Montanha processa a pergunta
    ├─ Gera resposta especializada
    └─ Retorna: {'resposta': 'Olá! Sou o Sr Montanha...'}
    ↓
[END]
    ↓
OUTPUT: Exibe a resposta do Sr Montanha
'''