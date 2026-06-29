import math
import json
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. GÖREV: Matematiksel Kasları Çalıştırma
def cosine_similarity(vec_a, vec_b):
    """
    İki sayı listesi (vektör) arasındaki kosinüs benzerliğini hesapla.
    İpucu: zip(vec_a, vec_b) kullanarak elemanları çarpıp toplayabilirsin.
    Formül: dot_product / (norm_a * norm_b)
    """
    dot_product = sum(a*b for a,b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a*a for a in vec_a))
    norm_b = math.sqrt(sum(b*b for b in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def main():
    # SDK ve Veritabanı Kurulumu
    config = Configuration(app_name="rag_practice")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    conn = sqlite3.connect("practice_rag.db")
    cursor = conn.cursor()
    
    # 2. GÖREV: SQL Kaslarını Çalıştırma
    # 'knowledge' adında bir tablo oluştur: id (primary key), term (text), definition (text), embedding (text) olsun.
    # ====== KODU BURAYA YAZ (GÖREV 2) ======

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS knowledge(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       term TEXT NOT NULL,
                       definition TEXT NOT NULL,
                       embedding TEXT NOT NULL
                   )
                   """)
    # =======================================
    conn.commit()

    # Modelleri Hazırlama
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()
    emb_client = embedding_model.get_embedding_client()

    # Örnek Veri Seti (Eğer tablo boşsa ekleyelim)
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    if cursor.fetchone()[0] == 0:
        print("Veritabanı boş, örnek terimler ekleniyor...")
        data = [
            ("Compiler", "Compilers translate high-level source code into machine code that CPUs can execute directly."),
            ("Database", "Databases are structured systems used to store, manage, and retrieve electronic data efficiently."),
            ("API", "Application Programming Interfaces allow different software applications to communicate with each other.")
        ]
        
        for term, definition in data:
            # Metnin embedding'ini alıyoruz
            emb_res = emb_client.generate_embeddings([definition])
            vector = emb_res.data[0].embedding
            
            # 3. GÖREV: Veritabanına Kaydetme
            # term, definition ve json'a çevrilmiş vector değerini tabloya INSERT et.
            # ====== KODU BURAYA YAZ (GÖREV 3) ======
            cursor.execute(
                """INSERT INTO knowledge (term,definition,embedding) VALUES (?,?,?)""",(term, definition, json.dumps(vector)))
            
            
            # =======================================
        conn.commit()
        print("Kayıtlar başarıyla eklendi!")

    # --- INTERAKTİF ARAMA TESTİ ---
    query = input("\nAratmak istediğiniz kavramı yazın (Örn: 'How code runs on CPU'): ")
    
    # Kullanıcının sorgusunu vektörleştiriyoruz
    query_emb = emb_client.generate_embeddings([query]).data[0].embedding

    # Veritabanındaki tüm verileri çekip anlamsal arama yapıyoruz
    cursor.execute("SELECT term, definition, embedding FROM knowledge")
    rows = cursor.fetchall()

    best_term = None
    best_def = None
    best_score = -1

    for row in rows:
        term, definition, emb_str = row
        db_vector = json.loads(emb_str) # JSON'dan geri listeye çeviriyoruz
        
        # Yazdığın benzerlik fonksiyonunu test ediyoruz
        score = cosine_similarity(query_emb, db_vector)
        
        if score > best_score:
            best_score = score
            best_term = term
            best_def = definition

    print(f"\n[Sonuç] En yakın terim: {best_term} (Benzerlik Skoru: {best_score:.4f})")
    print(f"Tanımı: {best_def}")

    embedding_model.unload()
    conn.close()

if __name__ == "__main__":
    main()