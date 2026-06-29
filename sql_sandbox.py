import sqlite3
import json

def main():
    db_name = "rag_sandbox.db"
    
    # 1. ADIM: Veritabanına Bağlanma ve Tablo Oluşturma
    # Dosya yoksa otomatik oluşturulur.
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    print(f"[1] '{db_name}' veritabanına bağlanıldı.")
    

    # Üstteki satırın doğrusunu altta çalıştırıyoruz:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()
    print("-> 'documents' tablosu oluşturuldu (id, content, embedding).")

    # 2. ADIM: Örnek Veri Ekleme (Insertion Practice)
    print("\n[2] Örnek veriler veritabanına ekleniyor...")
    
    # Simüle edilmiş basit embedding vektörleri (0.6b modelinden gelmiş gibi)
    sample_data = [
        ("Foundry Local runs AI models directly on your device.", [0.12, -0.43, 0.89]),
        ("The Foundry Local SDK supports Python and C#.", [0.55, 0.01, -0.22]),
        ("Embedding models convert text into numerical vectors.", [-0.09, 0.74, 0.11])
    ]
    
    for content, vector in sample_data:
        # Vektör listesini SQLite'ın anlayacağı TEXT (JSON string) formatına çeviriyoruz
        vector_json = json.dumps(vector)
        
        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (?, ?)",
            (content, vector_json)
        )
    
    conn.commit()
    print(f"-> {len(sample_data)} adet döküman satırı başarıyla kaydedildi.")

    # 3. ADIM: ID ile Kayıt Getirme (Retrieve by ID)
    print("\n[3] ID'ye göre kayıt sorgulama (ID: 2)...")
    cursor.execute("SELECT id, content, embedding FROM documents WHERE id = ?", (2,))
    row = cursor.fetchone()
    
    if row:
        fetched_id, fetched_content, fetched_embedding_str = row
        # Metin olarak aldığımız embedding'i tekrar Python listesine (float) çeviriyoruz
        fetched_vector = json.loads(fetched_embedding_str)
        
        print(f"-> Bulunan Kayıt:")
        print(f"   ID: {fetched_id}")
        print(f"   İçerik: {fetched_content}")
        print(f"   Vektör Tipi: {type(fetched_vector)} | Değer: {fetched_vector}")

    # 4. ADIM: Kelime ile Filtreleme (Filter by Text Keyword)
    keyword = "models"
    print(f"\n[4] Anahtar kelimeye göre döküman filtreleme (Kelime: '{keyword}')...")
    
    # SQL LIKE operatörü ile metin içinde arama yapıyoruz
    cursor.execute("SELECT id, content FROM documents WHERE content LIKE ?", (f"%{keyword}%",))
    results = cursor.fetchall()
    
    print(f"-> '{keyword}' kelimesini içeren sonuçlar:")
    for res_id, res_content in results:
        print(f"   * [ID: {res_id}] {res_content}")

    # Bağlantıyı kapatma
    conn.close()
    print("\nVeritabanı bağlantısı güvenli bir şekilde kapatıldı.")

if __name__ == "__main__":
    main()