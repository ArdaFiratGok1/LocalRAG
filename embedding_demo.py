import math
from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. Adım: Kosinüs Benzerliği Hesaplama Fonksiyonu
def cosine_similarity(vec_a, vec_b):
    """İki vektör arasındaki yönsel benzerliği (0 ile 1 arasında) hesaplar."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def main():
    # 2. Adım: Foundry Local SDK Başlatma
    config = Configuration(app_name="foundry_local_demo")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 3. Adım: Önerilen Küçük Embedding Modelini Yükleme
    print("Embedding modeli yükleniyor...")
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rModel İndirme Durumu: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # 4. Adım: Örnek Cümle Listesi (Knowledge Base)
    documents = [
        "Foundry Local runs AI models directly on your device without cloud connectivity.",
        "The Foundry Local SDK supports Python, C#, JavaScript, and Rust.",
        "Embedding models convert text into numerical vectors for similarity search.",
        "Retrieval-augmented generation grounds model responses in your own data.",
        "Vector similarity search finds documents that are semantically close to a query."
    ]

    print("\n[1/3] Belgeler vektör uzayına aktarılıyor (Embedding üretiliyor)...")
    # Tüm belgelerin embedding'lerini tek bir batch çağrısıyla üretiyoruz
    response = embedding_client.generate_embeddings(documents)
    doc_embeddings = [item.embedding for item in response.data]
    print(f"-> {len(doc_embeddings)} belge başarıyla vektörleştirildi.")

    # 5. Adım: Arama Sorgusu (Query) Tanımlama ve Vektörleştirme
    query = "Which programming languages can I use with the SDK?"
    print(f"\n[2/3] Kullanıcı sorgusu taranıyor: '{query}'")
    
    query_response = embedding_client.generate_embeddings([query])
    query_embedding = query_response.data[0].embedding

    # 6. Adım: Basit Döngü ile Benzerlik Skorlarını Hesaplama (Find Relevant)
    print("\n[3/3] Kosinüs Benzerliği hesaplanıyor ve en yakın eşleşme aranıyor...")
    
    best_score = -1
    best_match_index = -1

    for i, doc_vector in enumerate(doc_embeddings):
        # Her belgenin sorguya olan kosinüs benzerliğini ölçüyoruz
        score = cosine_similarity(query_embedding, doc_vector)
        print(f"   * Belge {i+1} Benzerlik Skoru: {score:.4f}")
        
        # En yüksek skora sahip olanı hafızada tutuyoruz
        if score > best_score:
            best_score = score
            best_match_index = i

    # 7. Adım: Sonucu Ekrana Basma
    print("\n================= EN YAKIN EŞLEŞME =================")
    print(f"En Alakalı Belge (İndis {best_match_index+1}):")
    print(f"-> \"{documents[best_match_index]}\"")
    print(f"Benzerlik Skoru: {best_score:.4f}")
    print("====================================================")

    # Temizlik
    embedding_model.unload()

if __name__ == "__main__":
    main()