import requests
import configparser
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag_system import RAG

rag = RAG()

# embeddings_model = "Sahajtomar/french_semantic"
# embedder = HuggingFaceEmbeddings(model_name=embeddings_model,cache_folder="./cache_folder")
# conf = configparser.ConfigParser()
# env = conf.read("RAG.ini")

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser') 
    
    # Extract text content from the page
    content = ' '.join([p.get_text() for p in soup.find_all('p')])
    
    return content


def scrape_website(base_url, max_pages=10):
    visited = set()
    to_visit = [base_url]
    scraped_data = {}

    while to_visit and len(scraped_data) < max_pages:
        url = to_visit.pop(0)
        if url not in visited:
            print(f"Scraping: {url}")
            content = scrape_page(url)  # parcours et télécharge toutes les pages du site web
            scraped_data[url] = content
            visited.add(url)
            # Find more links on the page
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                full_url = urljoin(base_url, link['href'])
                if full_url.startswith(base_url) and full_url not in visited:
                    to_visit.append(full_url)

    return scraped_data


def chunk_data(base_url):
    scraped_data = scrape_website(base_url)
    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=20,
    )
    for _, content in scraped_data.items():    
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk)
            doc_chunks.append(doc)
    return doc_chunks


def store_chunk(base_url):
    data = chunk_data(base_url)
    print("Saving embeddings in progress ...")
    db = Chroma.from_documents(documents=data, embedding=rag.embedder, persist_directory=rag.persist_directory)  # Sauvegarde du vectorStore
    print("Saving embeddings in progress ...")
    return db

vectStore = store_chunk(rag.url)

# vectStore = Chroma(rag.persist_directory, embedding_function=rag.embedder)  # Chargement du vectorStore
query = "quelles sont les formations dispenser au DIT?"
matched_docs = vectStore.similarity_search(query,k=5)  # Récupération des similarités
context = "\n".join([doc.page_content for doc in matched_docs]) # Définition du context à fournir au llm
print(context)