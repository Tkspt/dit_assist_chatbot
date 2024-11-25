import configparser
from dotenv import load_dotenv 
import os
from langchain_community.vectorstores import Chroma
from langchain.chains import  LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
load_dotenv()
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
class RAG:
    def __init__(self):
        self.CONFIG_FILE="RAG.ini"
        self.conf = configparser.ConfigParser()
        self.conf.read(self.CONFIG_FILE)
        self.embeddings_model = self.conf['RAG']['embedding_model']
        self.persist_directory = self.conf['RAG']['persist_directory']
        self.cache_folder = self.conf['RAG']['cache_folder']
        self.llm_model =  self.conf['RAG']['llm_model_path']
        # Load french embedding model and keep it in cache
        self.embedder = HuggingFaceEmbeddings(
        model_name=self.embeddings_model,
        cache_folder="./cache_folder",
        model_kwargs = {'device': 'cpu'})
        # Load opensource LLM model
        #self.llm = LlamaCpp(self.llm_model)
        # Charger le modèle Mistral
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        from accelerate.big_modeling import disk_offload
        self.offload_dir = "path/to/offload/directory"
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_model)
        disk_offload(self.model, self.offload_dir)

    def similarity_search(self, query):
     
        vectordb = Chroma(persist_directory="./chroma_db", embedding_function=self.embedder)
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
        db = Chroma(persist_directory="./chroma_db", embedding_function=self.embedder)
        matched_docs = db.similarity_search(query,k=5)
        context = "\n".join([doc.page_content for doc in matched_docs])
    
        # Créer un prompt avec le contexte et la question
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)

        # Tokenisation du prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

        # Génération de la réponse avec le modèle Mistral
        outputs = self.model.generate(**inputs,max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
rag = RAG()
#query = "quelles sont les formations dispenser au DIT?"
#response = rag.retrieve_response(query)
#print(response)
   