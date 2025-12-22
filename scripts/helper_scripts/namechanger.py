from pathlib import Path

# --- AYARLAR ---
TARGET_FOLDER = "./sampled_audio1"  # İşlem yapılacak klasörün yolu
TEXT_TO_APPEND = "_3"         # Eklenecek yazı (örn: _v1, _backup)

def main():
    folder = Path(TARGET_FOLDER)

    # Klasörün varlığını kontrol et
    if not folder.exists():
        print(f"Hata: '{TARGET_FOLDER}' klasörü bulunamadı.")
        return

    # Klasör içindeki dosyaları döngüye al
    for file_path in folder.iterdir():
        # Sadece dosyaları işle (alt klasörleri atla)
        if file_path.is_file():
            
            # Yeni ismi oluştur:
            # .stem   -> Dosyanın uzantısız adı (image)
            # .suffix -> Dosyanın uzantısı (.png)
            new_filename = f"{file_path.stem}{TEXT_TO_APPEND}{file_path.suffix}"
            
            # Yeni tam yolu oluştur
            new_file_path = file_path.with_name(new_filename)
            
            try:
                # Dosyayı yeniden adlandır
                file_path.rename(new_file_path)
                print(f"✅ Tamam: {file_path.name} -> {new_filename}")
            except Exception as e:
                print(f"❌ Hata: {file_path.name} değiştirilemedi. Sebep: {e}")

if __name__ == "__main__":
    main()