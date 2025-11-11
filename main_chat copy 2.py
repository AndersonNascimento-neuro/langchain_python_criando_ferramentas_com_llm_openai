#################################################################
# Simulando uma interação de chat com memória                   #
#################################################################

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Inicializa o modelo de chat da OpenAI
modelo = ChatOpenAI(
    model='gpt-3.5-turbo',  # Modelo GPT-3.5 Turbo
    temperature=0.5,  # Temperatura média para respostas equilibradas
    api_key=api_key  # Chave da API
)

# Define o template de prompt com três componentes:
# 1. Mensagem do sistema: define o papel e comportamento do assistente
# 2. Placeholder para histórico: onde as mensagens anteriores serão inseridas
# 3. Mensagem humana: a pergunta atual do usuário
prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ('system', 'Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios'),
        ('placeholder', '{historico}'),  # Placeholder que será substituído pelo histórico da conversa
        ('human', '{query}')  # Pergunta atual do usuário
    ]
)

# Cria a cadeia de processamento:
# prompt_sugestao → modelo → StrOutputParser (converte resposta em string)
cadeia = prompt_sugestao | modelo | StrOutputParser()

# Dicionário que armazenará o histórico de mensagens por sessão
# Cada sessão terá seu próprio histórico independente
memoria = {}

# ID da sessão atual - identifica unicamente esta conversa
sessao = 'aula_langchain_alura'

# Função que retorna o histórico de uma sessão específica
# Se a sessão não existe, cria um novo histórico vazio
def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        # Cria um novo histórico em memória para esta sessão
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

# Lista de perguntas que serão feitas sequencialmente
# A segunda pergunta faz referência ao contexto da primeira
lista_perguntas = [
    'Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?',
    'Qual a melhor época do ano para ir?'  # ✅ Agora o modelo entenderá o contexto
]

# Envolve a cadeia com funcionalidade de memória
# Isso permite que o modelo "lembre" das mensagens anteriores
cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,  # A cadeia original
    get_session_history=historico_por_sessao,  # Função que retorna o histórico
    input_messages_key='query',  # Nome da chave que contém a pergunta do usuário
    history_messages_key='historico'  # Nome da chave onde o histórico será inserido no prompt
)

# Itera sobre cada pergunta da lista
for pergunta in lista_perguntas:
    # Invoca a cadeia com memória, passando a pergunta e o ID da sessão
    resposta = cadeia_com_memoria.invoke(
        {
            'query': pergunta  # A pergunta atual
        },
        config={
            'session_id': sessao  # Identifica qual sessão usar (qual histórico carregar)
        }
    )
    
    # Exibe a pergunta do usuário
    print(f'Usuário: {pergunta}')
    
    # Exibe a resposta do modelo (que agora tem contexto das mensagens anteriores)
    print(f'IA: {resposta}\n')
    
# ✅ SOLUÇÃO: Agora o modelo mantém o contexto entre as perguntas
# O histórico é automaticamente atualizado após cada interação:
# - Pergunta do usuário é adicionada ao histórico
# - Resposta da IA é adicionada ao histórico
# - Próxima pergunta terá acesso a todo o contexto anterior