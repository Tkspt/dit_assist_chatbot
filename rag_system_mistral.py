import configparser
from dotenv import load_dotenv 
import os
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

class MISTRAL_RAG:
    def __init__(self):
        self.CONFIG_FILE = "RAG.ini"
        self.conf = configparser.ConfigParser()
        self.conf.read(self.CONFIG_FILE)

        # Charger les configurations
        self.embeddings_model = self.conf['RAG']['embedding_model']
        self.persist_directory = self.conf['RAG']['persist_directory']
        self.cache_folder = self.conf['RAG']['cache_folder']
        self.llm_model_path = self.conf['RAG']['mistral_llm_model_path']

        print("Loading embedding model in progress ...")
        # Charger le modèle d'embeddings
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            cache_folder=self.cache_folder,
            model_kwargs={'device': 'cpu'}
        )
        print("Loading embedding model done")

        # Charger le modèle Mistral
        print("Loading Mistral model in progress ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_model_path, device_map="auto")
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)  # To run it on GPU use device=0 ...
        print("Loading Mistral model done")

    def similarity_search(self, query, k=5):
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder)
        return vectordb.similarity_search(query, k=k)
    
    def retrieve_response(self, query):
        # print("e")
        template = """
          Vous êtes Aida, un agent conversationnel qui ne répond que sur des questions relatives aux formations dispensées au Dakar Institut of Technology (DIT).
          - Merci de ne pas mentionner le fait que vous êtes une IA.
          Context: {context}
          ---
          Question: {question}
          Answer:
        """
        print(query)
        
        matched_docs = self.similarity_search(query)
        context = "\n".join([doc.page_content for doc in matched_docs])
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
        
        # Générer la réponse avec Mistral
        full_prompt = prompt.format(question=query)
        
        response = self.pipeline(full_prompt, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']

        return response
    

        # Créer un prompt avec le contexte et la question
        # prompt = template.format(context=context, question=query)

        # Tokenisation du prompt
        # inputs = self.tokenizer(full_prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

        # Génération de la réponse avec le modèle Mistral
        # outputs = self.model.generate(**inputs, max_length=500)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return response

# Exemple d'utilisation
rag = MISTRAL_RAG()
query = "Quelles sont les formations dispensées au DIT ?"
response = rag.retrieve_response(query)
print(response)
