import configparser
from dotenv import load_dotenv 
import os
from langchain_community.vectorstores import Chroma
from langchain.chains import  LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class RAG:
    def __init__(self):
        self.CONFIG_FILE="RAG.ini"
        self.conf = configparser.ConfigParser()
        self.conf.read(self.CONFIG_FILE)

        # Charger les configurations
        self.embeddings_model = self.conf['RAG']['embedding_model']
        self.url = self.conf['RAG']['url']
        self.persist_directory = self.conf['RAG']['persist_directory']
        self.cache_folder = self.conf['RAG']['cache_folder']

        #self.encoding_model =  self.conf['RAG']['encoding_model']
        self.llm_model =  self.conf['RAG']['open_ai_llm_model_path']

        print("Loading embedding model in progress ...")
        # Load french embedding model and keep it in cache
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            cache_folder=self.cache_folder,
            model_kwargs = {'device': 'cpu'}
        )
        print("Loading embedding model done")

        # Load opensource LLM model
        #self.llm = LlamaCpp(self.llm_model)

    def similarity_search(self, query):
     
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder)
        result = vectordb.similarity_search(query)
       
        return result
    
    def retrieve_response(self,query):
        template = """
          Vous êtes Aida, un agent  conversationel qui ne répond que sur des questions relatives  au formation dispensé au Dakar  Institut of Technology (DIT)
          - Merci de ne pas mentionner le fait que vous êtes AI 
           Context: {context}
           ---
           Question: {question}
           Answer:"
        """
        print(query)
        # db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder)
        # matched_docs = db.similarity_search(query,k=5)

        matched_docs = self.similarity_search(query)
        context = "\n".join([doc.page_content for doc in matched_docs])
    
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
        llm_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(model_name=self.llm_model,temperature=1.0))

        return llm_chain.run(query)
    

rag = RAG()
query = "quelles sont les formations dispenser au DIT?"
response = rag.retrieve_response(query)
print(response)
   