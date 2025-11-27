######################################################
# Usando RAG em um txt com LangChain                 #
######################################################

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Inicializa o modelo de chat da OpenAI
model = ChatOpenAI(
    model='gpt-4o-mini',  # Modelo GPT-4o-mini
    temperature=0.5,  # Temperatura média
    api_key=api_key
)

# Inicializa o modelo de embeddings da OpenAI
# Embeddings são representações vetoriais de texto usadas para busca semântica
embeddings = OpenAIEmbeddings()

# Carrega o documento de texto do arquivo especificado
# TextLoader lê o arquivo e prepara para processamento
documento = TextLoader(
    'documentos/GTB_gold_Nov23.txt',  # Caminho do arquivo (parece ser um documento de seguro)
    encoding='utf-8'  # Codificação do arquivo
)

# Divide o documento em PEDAÇOS menores (chunks)
# RecursiveCharacterTextSplitter quebra o texto de forma inteligente
pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Cada pedaço terá no máximo 1000 caracteres
    chunk_overlap=100  # Sobreposição de 100 caracteres entre pedaços (para não perder contexto)
).split_documents(documento.load())

# Cria um VECTOR STORE (banco de dados vetorial) usando FAISS
# FAISS armazena os embeddings dos pedaços para busca eficiente
# O método as_retriever() transforma o vector store em um recuperador
dados_recuperados = FAISS.from_documents(
    pedacos,  # Os pedaços do documento
    embeddings  # O modelo de embeddings para vetorizar os pedaços
).as_retriever(search_kwargs={'k': 2})  # Retorna os 2 pedaços mais relevantes por busca

# Define o prompt para consulta ao seguro
# Este prompt instrui o modelo a usar APENAS o contexto fornecido
prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ('system', 'Responda usando exclusivamente o conteúdo fornecido'),  # Restrição importante
        ('human', '{query}\n\nContexto: \n{contexto}\n\nResposta:')  # Formato: pergunta + contexto
    ]
)

# Cria a cadeia de processamento RAG (Retrieval-Augmented Generation)
# prompt → modelo → parser (string)
cadeia = prompt_consulta_seguro | model | StrOutputParser()

# Função principal que implementa o padrão RAG
def responder(pergunta: str):
    # PASSO 1: RECUPERAÇÃO (Retrieval)
    # Busca os pedaços mais relevantes do documento usando busca semântica
    trechos = dados_recuperados.invoke(pergunta)
    
    # PASSO 2: PREPARAÇÃO DO CONTEXTO
    # Concatena os pedaços recuperados em uma única string
    # page_content contém o texto de cada pedaço
    contexto = '\n\n'.join(um_trecho.page_content for um_trecho in trechos)
    
    # PASSO 3: GERAÇÃO (Generation)
    # Envia a pergunta + contexto para o modelo gerar a resposta
    return cadeia.invoke(
        {
            'query': pergunta,  # A pergunta do usuário
            'contexto': contexto  # O contexto recuperado do documento
        }
    )
    
# Testa a função com uma pergunta sobre procedimento de roubo
print(responder('Como devo proceder caso tenha um item roubado?'))


## **Como funciona o padrão RAG (Retrieval-Augmented Generation):**
'''
PERGUNTA: "Como devo proceder caso tenha um item roubado?"
    ↓
[PASSO 1: RECUPERAÇÃO]
    ├─ Converte pergunta em embedding (vetor)
    ├─ Busca os 2 pedaços mais similares no FAISS
    └─ Retorna trechos relevantes do documento de seguro
    ↓
[PASSO 2: PREPARAÇÃO]
    ├─ Concatena os trechos recuperados
    └─ Cria o contexto: "Seção 5.2: Em caso de roubo..."
    ↓
[PASSO 3: GERAÇÃO]
    ├─ Envia pergunta + contexto para o GPT-4o-mini
    ├─ Modelo gera resposta baseada APENAS no contexto
    └─ "De acordo com o documento, você deve..."
    ↓
OUTPUT: Resposta fundamentada no documento
'''