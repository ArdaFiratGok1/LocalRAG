import os
import sqlite3
import json
import math
from foundry_local_sdk import Configuration, FoundryLocalManager

def load_and_split(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return []
    files = os.listdir(folder_path)
    chunks = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                paragraphs = content.split("\n")

                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph and not paragraph.startswith("---") and len(paragraph) > 3:
                        chunks.append({ "source": file, "content": paragraph })


    return chunks

def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def find_relevant(query_embedding, db_rows, top_k=2):
    scores = []
    for row in db_rows:
        doc_id, text, emb_str = row
        doc_emb = json.loads(emb_str) # Veritabanından gelen string'i listeye çeviriyoruz
        
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((text, score)) # İndis yerine direkt metni (text) saklıyoruz
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def main():


    config = Configuration(app_name="localRAG")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Load the embedding model
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    total_chunks = load_and_split(r"C:\Users\arda fırat\Desktop\rag_testfiles")

    response = embedding_client.generate_embeddings([chunk['content'] for chunk in total_chunks])
    chunk_embeddings = [item.embedding for item in response.data]
    print(f"Indexed {len(chunk_embeddings)} documents.")

    db_name = "RAGSqlite.db"

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    '''
    cursor.execute("DROP TABLE IF EXISTS documents")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   text TEXT NOT NULL,
                   embedding TEXT NOT NULL
        )
    """)

    conn.commit()

    for chunk,embedding in zip(total_chunks, chunk_embeddings):
        cursor.execute("INSERT INTO documents (text, embedding) VALUES (?, ?)", (chunk['content'], json.dumps(embedding)))

    conn.commit()
    '''
    cursor.execute("SELECT id, text, embedding FROM documents")
    rows = cursor.fetchall()
    print(f"Retrieved {len(rows)} documents.")

    print()
    query = input("Enter your query: ")
    response = embedding_client.generate_embeddings([query])
    query_embedding = response.data[0].embedding
    
    
    results = find_relevant(query_embedding, rows, top_k=3)

    print("Relevant documents:")
    for text, score in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {text}")
        print("-" * 30)
    print("="*40)

    embedding_model.unload()
    conn.close()

if __name__ == "__main__":
    main()

