from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    config = Configuration(app_name="katalog_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Foundry Local Kataloğundaki Modeller Listeleniyor...\n")
    
    models = manager.catalog.list_models()
    
    # İlk modelin içindeki tüm gizli/açık özellikleri terminale basalım ki kopya çekebilelim
    if models:
        print("--- Geliştirici Notu: Model Nesnesinin İç yapısı ---")
        print(dir(models[0]))
        print("-" * 50 + "\n")
    
    # Şimdi sadece hata vermeyecek temel bilgileri ekrana basalım
    for m in models:
        # alias özelliğinin varlığından eminiz, yanına m.name veya m.id gibi alternatifleri de deneyebilirsin
        print(f"Model: {m.alias}")

if __name__ == "__main__":
    main()