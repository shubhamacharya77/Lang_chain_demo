from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 
from dotenv import load_dotenv
load_dotenv()


#llm 
model= ChatOllama(
    model="phi3:mini"
)
model2 = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#pdf loader 
loader=DirectoryLoader("docs",glob="*.pdf",loader_cls=PyPDFLoader)
data=loader.load()


# text splitter 
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks=splitter.split_documents(data)

#embedding model
# embedding_model = GoogleGenerativeAIEmbeddings( model="gemini-embedding-001",api_key=os.getenv("GEMINI_API_KEY"))

# store in 
if os.path.exists("./vector_db"):
    db = Chroma(
        persist_directory="./vector_db",
        embedding_function=model2     #embedding_model
    )
else:
    db = Chroma.from_documents(
        documents=chunks,
        embedding=model2,      #embedding_model,
        persist_directory="./vector_db"
    )



user_input= input("enter your query :")

user_input_embedding=model2.embed_query(user_input)

contexts=db.similarity_search_by_vector(
    embedding=user_input_embedding,
    k=3
)
docs=[context.page_content for context in contexts]


#user phase
parser=StrOutputParser()
propmt=PromptTemplate(
    template="""
You are a helpful AI assistant that answers questions ONLY using the information available in the provided documents/context.

Rules:
1. Carefully read the provided document context before answering.
2. If the user's question can be answered from the document, give a clear, accurate, and concise response based only on that information.
3. Do not use outside knowledge, assumptions, or hallucinations.
4. If the answer is partially available, state what is available and mention what is missing.
5. If the answer is not found in the provided documents, politely respond:

"I'm sorry, but I could not find that information in the provided documents."
6. Keep the tone professional, polite, and helpful.
7. If relevant, summarize long answers in simple language.
8. dislay the context which provided to you.
Context:
{context}

User Question:
{user_input}

Answer:

""",
input_variables=["user_input","context"]
)

# print(response)
chain=propmt|model|parser
response=chain.invoke({
"user_input":user_input,
"context":"\n\n".join(docs)
}
)
print(response)



