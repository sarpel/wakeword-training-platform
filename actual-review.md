### 🤖 Kodlama Ajanı İçin Talimat (Prompt)
**Önemli Not:** Aşağıdaki analiz `src/models/architectures.py`, `src/training/trainer.py` ve `src/ui/panel_training.py` dosyalarının mevcut haline dayanmaktadır. Config parametrelerinin koda aktarılmadığı ("Silent Bug") tespit edilmiştir.
**Görev:** Aşağıdaki 3 kritik dosyayı güncelleyerek, `config.yaml` dosyasındaki tüm gelişmiş parametrelerin (Bidirectional, RNN layers, Time Shift vb.) eğitim sürecine dahil edilmesini sağla.
#### 1. Dosya: `src/models/architectures.py` (En Kritik Eksiklik)
Şu anki kodda `create_model` fonksiyonu, `kwargs` içindeki parametreleri `MobileNetV3Wakeword` ve `TinyConvWakeword` sınıflarına **göndermiyor**. Ayrıca bu sınıflar gönderilse bile bu parametreleri kullanacak yapıya sahip değil.
* **Düzeltme 1 (`create_model`):**
* `kwargs` sözlüğünü olduğu gibi model sınıflarına ilet.
* *Mevcut:* `dropout=kwargs.get("dropout", 0.3)`
* *İstenen:* `**kwargs` ekle.
* **Düzeltme 2 (`MobileNetV3Wakeword` Sınıfı):**
* `__init__` metoduna `hidden_size`, `num_layers` (RNN için), `bidirectional` argümanlarını ekle.
* Eğer `num_layers > 0` ise, MobileNet'in özellik çıkarıcısı (`self.mobilenet.features`) ile sınıflandırıcı (`self.mobilenet.classifier`) arasına bir **LSTM veya GRU** katmanı ekle.
* *Mantık:* Config dosyasında `bidirectional: true` seçildiyse, model sadece CNN değil, CNN+LSTM (CRNN) gibi davranmalı.
* **Düzeltme 3 (`TinyConvWakeword` Sınıfı):**
* `__init__` metoduna `tcn_num_channels` listesini parametre olarak ekle.
* Sabit yazılmış (16, 32, 64, 64) katman yapısını sil.
* Bunun yerine, `tcn_num_channels` listesi üzerinde döngü kurarak `self.features` katmanlarını dinamik olarak oluştur. Böylece kullanıcı config dosyasından modelin derinliğini ve genişliğini kontrol edebilir.
#### 2. Dosya: `src/training/trainer.py` (Bağlantı Kopukluğu)
`Trainer` sınıfı başlatılırken `create_model` fonksiyonunu çağırıyor ancak config dosyasındaki gelişmiş model ayarlarını parametre olarak geçmiyor.
* **Düzeltme:**
* `Trainer.__init__` içindeki `create_model` çağrısını güncelle.
* Aşağıdaki parametreleri config'den alıp fonksiyona ekle:
```python
# create_model çağrısına eklenecekler:
hidden_size=config.model.hidden_size,
num_layers=config.model.num_layers,
bidirectional=config.model.bidirectional,
tcn_num_channels=getattr(config.model, "tcn_num_channels", None),
tcn_kernel_size=getattr(config.model, "tcn_kernel_size", 3),
# Diğer tüm kwargs...
```
#### 3. Dosya: `src/ui/panel_training.py` (Veri Kaybı)
Eğitimi başlatan `start_training` fonksiyonu, `aug_config` sözlüğünü oluştururken yeni eklenen "Time Shift" (Zaman Kaydırma) özelliğini unutuyor.
* **Düzeltme:**
* `aug_config` sözlüğüne şunları ekle:
```python
"time_shift_prob": getattr(config.augmentation, "time_shift_prob", 0.0),
"time_shift_range_ms": (
    getattr(config.augmentation, "time_shift_min_ms", -100),
    getattr(config.augmentation, "time_shift_max_ms", 100),
),
```
* Bu yapılmazsa, `AudioAugmentation` sınıfı varsayılan değerleri kullanır ve config ayarları boşa gider.
