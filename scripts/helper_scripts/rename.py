import os


def toplu_yeniden_adlandir(klasor_yolu, yeni_ad):
    # Klasördeki tüm dosyaları listele
    dosyalar = os.listdir(klasor_yolu)

    # Dosyaları alfabetik sıraya diz (düzenli olması için)
    dosyalar.sort()

    # Sayaç ile döngüye gir (index 0'dan başlar, o yüzden sayi+1 kullanacağız)
    for index, eski_dosya_adi in enumerate(dosyalar):
        # Tam dosya yolunu oluştur
        eski_yol = os.path.join(klasor_yolu, eski_dosya_adi)

        # Eğer bu bir klasörse atla, sadece dosyaları değiştir
        if not os.path.isfile(eski_yol):
            continue

        # Dosya uzantısını al (örn: .jpg, .png)
        dosya_adi, uzantı = os.path.splitext(eski_dosya_adi)

        # Yeni ismi formatla:
        # {yeni_ad} : senin belirlediğin isim
        # {index + 1:06d} : 1'den başla, 6 haneye tamamla (000001)
        yeni_dosya_adi = f"{yeni_ad}_{index + 1:06d}{uzantı}"

        # Yeni tam yolu oluştur
        yeni_yol = os.path.join(klasor_yolu, yeni_dosya_adi)

        # Dosya ismini değiştir
        os.rename(eski_yol, yeni_yol)
        print(f"Değiştirildi: {eski_dosya_adi} -> {yeni_dosya_adi}")


# --- KULLANIM ---
# Buradaki yolu kendi klasör yolunla değiştir
hedef_klasor = "./rirs"
# Vermek istediğin temel ad
belirlenen_ad = "rirs"

toplu_yeniden_adlandir(hedef_klasor, belirlenen_ad)
