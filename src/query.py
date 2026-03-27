import os
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

# --- CONFIGURAZIONE ---
# Usiamo l'endpoint di Ollama che espone API compatibili con OpenAI
client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama")
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "vectordb"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "secret"),
    "port": os.getenv("DB_PORT", "5432")
}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

def get_query_embedding(text):
    """Trasforma la domanda dell'utente in un vettore (Embedding)."""
    response = client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def search_context(query_text, limit=3):
    """Cerca nel DB i frammenti più simili alla domanda."""
    query_vector = get_query_embedding(query_text)
    
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    # Operatore <=> calcola la distanza del coseno (più è piccola, più sono simili)
    cur.execute("""
        SELECT content, filename, page_number
        FROM document_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_vector, limit))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def ask_elephant(question):
    """Esegue il processo RAG: Recupero + Generazione."""
    
    # 1. RETRIEVAL: Trova i pezzi di PDF che contengono la risposta
    relevant_chunks = search_context(question)
    
    if not relevant_chunks:
        return "Non ho trovato informazioni pertinenti nel database."

    # Uniamo i frammenti in un unico blocco di testo (Contesto)
    context_text = "\n\n".join([
        f"[Fonte: {res[1]}, Pagina {res[2]}]: {res[0]}" 
        for res in relevant_chunks
    ])

    # 2. AUGMENTATION: Prepariamo il prompt per l'LLM
    system_prompt = (
        "Sei un assistente tecnico esperto. Rispondi alla domanda dell'utente "
        "usando ESCLUSIVAMENTE il contesto fornito dai documenti qui sotto. "
        "Se la risposta non è presente nel contesto, dì onestamente che non lo sai."
    )
    
    user_prompt = f"CONTESTO:\n{context_text}\n\nDOMANDA: {question}"

    # 3. GENERATION: Chiamata a Llama 3
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2, # Bassa temperatura = risposta più precisa e meno creativa
        stream=True
    )
    
    #return response.choices[0].message.content
    return response

if __name__ == "__main__":
    print("🐘 Elephant-Mind Ready. Digita 'esci' per terminare.")
    while True:
        query = input("\nDomanda: ")
        if query.lower() in ['esci', 'quit', 'exit']:
            break
            
        print("\n🔍 Ricerca nel DB e analisi in corso...")
        risposta = ask_elephant(query)

        #print(f"\n🤖 RISPOSTA:\n{risposta}")
        print("\n🤖 RISPOSTA:")
        for chunk in risposta:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")