import fitz  # PyMuPDF
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import os
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

# --- CONFIGURAZIONE ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "vectordb"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "secret"),
    "port": os.getenv("DB_PORT", "5432")
}

# Inizializza client stile OpenAI per LM Studio / Ollama
client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

def extract_and_chunk(pdf_path, chunk_size=1000, overlap=100):
    """Estrae testo dal PDF e lo divide in pezzi sovrapposti."""
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        # Dividiamo il testo della pagina in blocchi
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append({
                "page": page_num + 1,
                "content": chunk.strip()
            })
    return chunks

def get_embedding(text):
    """Chiede l'embedding a ollama"""
    response = client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL 
    )
    return response.data[0].embedding

def save_to_db(pdf_name, chunks):
    """Connette a Postgres e salva i vettori."""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    # Creazione tabella se non esiste (768 è la dimensione di nomic-embed)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id serial PRIMARY KEY,
            filename text,
            page_number int,
            content text,
            embedding vector(768)
        );
    """)

    print(f"Inserimento di {len(chunks)} frammenti nel database...")
    
    for chunk in chunks:
        vector = get_embedding(chunk["content"])
        cur.execute("""
            INSERT INTO document_embeddings (filename, page_number, content, embedding)
            VALUES (%s, %s, %s, %s)
        """, (pdf_name, chunk["page"], chunk["content"], vector))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Ingestione completata con successo!")

if __name__ == "__main__":
    PDF_FILE = "data/pdf/gdpr.pdf" # Assicurati che il file esista
    
    if os.path.exists(PDF_FILE):
        print(f"Inizio elaborazione: {PDF_FILE}")
        data_chunks = extract_and_chunk(PDF_FILE)
        save_to_db(os.path.basename(PDF_FILE), data_chunks)
    else:
        print(f"Errore: Il file {PDF_FILE} non è stato trovato.")