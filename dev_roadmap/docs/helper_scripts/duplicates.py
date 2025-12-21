import os
import hashlib
import shutil

# Dosyanın MD5 hash'ini hesaplayan fonksiyon
def calculate_md5(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            # Dosyayı parça parça oku (RAM'i şişirmemek için)
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (OSError, IOError):
        return None

def find_and_move_duplicates():
    # Betiğin çalıştığı mevcut klasörü al
    root_dir = os.getcwd()
    target_folder = os.path.join(root_dir, "duplicate")
    
    # "duplicate" klasörü yoksa oluştur
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Klasör oluşturuldu: {target_folder}")

    # Görülen hashleri ve dosya yollarını saklayacağımız sözlük
    seen_hashes = {}
    
    # Tüm alt klasörleri gez
    for dirpath, _, filenames in os.walk(root_dir):
        
        # Hedef klasörün kendisini taramayı atla
        if os.path.abspath(dirpath) == os.path.abspath(target_folder):
            continue

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            
            # Betiğin kendisini (script.py) taşıma
            if filepath == os.path.abspath(__file__):
                continue

            file_hash = calculate_md5(filepath)
            
            if file_hash:
                if file_hash in seen_hashes:
                    # Eşleşme bulundu, bu bir kopya dosya
                    original_file = seen_hashes[file_hash]
                    print(f"[DUPLICATE] Bulundu: {filepath}")
                    print(f"            Orijinali: {original_file}")
                    
                    # Dosyayı taşı
                    try:
                        # Eğer hedefte aynı isimde dosya varsa üzerine yazmaması için isim değiştirilebilir
                        # (Burada basitlik adına direkt taşıma yapıyoruz)
                        dst_path = os.path.join(target_folder, filename)
                        
                        # Hedefte aynı isim varsa çakışmayı önlemek için basit rename
                        if os.path.exists(dst_path):
                            base, ext = os.path.splitext(filename)
                            dst_path = os.path.join(target_folder, f"{base}_copy{ext}")

                        shutil.move(filepath, dst_path)
                        print(f"            Taşındı -> {dst_path}")
                    except Exception as e:
                        print(f"            Hata oluştu: {e}")
                else:
                    # Yeni hash, sözlüğe ekle
                    seen_hashes[file_hash] = filepath

if __name__ == "__main__":
    print("Tarama başlıyor...")
    find_and_move_duplicates()
    print("İşlem tamamlandı.")