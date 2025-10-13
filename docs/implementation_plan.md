## 1) Veri + Özellik

* **CMVN kapsamı:** corpus-geneli CMVN + per-utterance değil. `stats.json` kaydet, eğitim/val/test aynı istatistiği kullansın.
* **Speaker-stratified split:** konuşmacı bazlı K-fold ve train/val/test ayrımı zorunlu. Hash = `sha1(path|duration|speaker_id)`.
* **BalancedBatchSampler:** her minibatch’te {pos, neg, hard_neg} oranını sabitle (ör. 1:1:1 veya 1:2:1).
* **Hard-neg mining planı:**

  * Pass-1: mevcut veriyle eğit.
  * Pass-2: val/test dışı uzun kayıtları kaydır pencerelerle tara → skor>thr ve “negative” klasörden gelenleri **HN pool**’a ekle.
  * Pass-3: her epoch minibatch’lerinin %25–35’i HN’den.

## 2) Eğitim döngüsü

* **EMA:** `decay=0.999` başla, son 10 epoch’ta `0.9995`. Sadece val ölçerken EMA ağırlıklarını uygula.
* **Grad-norm log:** her epoch `total_norm` yaz; `>5× median` ise `clip=1.0`.
* **LR-finder:** 100–200 adımda `1e-6→1e-2` log-sweep; en düşük `loss/lr` eğim kırılmasından **lr≈3e-4** doğrula.
* **Ablation flags:** `--no-rir`, `--no-specaug`, `--no-mixbg` CLI ile; tek tuş karşılaştırma.

## 3) Kayıp + Kalibrasyon

* **İlk eğitim:** `cross_entropy + class_weights=balanced + label_smoothing=0.05`.
* **Focal’a geçiş koşulu:** val’de **FN yüksek** ve FA düşükse → `focal(alpha=0.25, gamma=2.0)`, smoothing=0, weights=none.
* **Temperature scaling (val set):**

  * `T` için `minimize NLL( logits/T, labels )` tek skaler; ardından inference: `softmax(logits/T)`.
* **Operating point:** hedef **FAH**.

  * `FAH = FP / (Total_seconds/3600)`
  * Grid: `thr ∈ [0,1], step=0.0025`. `FAH<=target` koşulunu sağlayan en yüksek **TPR**’lı eşiği seç.

## 4) Metrikler

* **FAH, MR@FAH, FPR@TPR=0.95, EER, pAUC** hesapla.
* **Confusion matrisi** ve **DET eğrisi** sakla.
* **Latency:** `N=1000` ileri besleme, ortalama ve p95; batch=1, `torch.no_grad()`.

## 5) Doğrulama protokolü

* **Speaker K-fold (k=5)** raporu zorunlu.
* **Domain shift suite:** {8/16 kHz, SNR {0,5,10,20} dB, device {phone, laptop, mic}} × 200 örnek.
* **Reproducibility:** `seed=42`, `cudnn.deterministic=True`, konfigürasyon hash’ini modele göm.

## 6) Çıkarım + Ürünleştirme

* **Sliding window:** `win=1.0s`, `hop=0.1s`.
* **Voting + hysteresis:** `vote=3/5`, `on_thr=thr`, `off_thr=thr-0.1`, `lockout_ms=1500`.
* **TTA (test-time):** time-shift {−40, −20, 0, +20, +40 ms} skor ortalaması.
* **Export:** ONNX opset 17, dynamic axes `{batch:0, time:2}`; PTQ INT8 dene.
* **Model card:** sürüm, veri özetleri, CMVN stats, eşik, FAH@eşik, latency.

## 7) IO + Önbellek

* **FeatureCache (LRU):** Sondaki Örneklere göre seç.
* **mmap:** `np.load(..., mmap_mode='r')`.
* **DataLoader:** train `num_workers=16`, val `=8`, `pin_memory=True`, `prefetch_factor={4,2}`, `persistent_workers=True`.

### Kısa kod ekleri

**BalancedBatchSampler iskeleti**

```python
class BalancedBatchSampler(Sampler):
    def __init__(self, idx_pos, idx_neg, idx_hn, batch_size):
        self.b = batch_size//3
        self.pos, self.neg, self.hn = idx_pos, idx_neg, idx_hn
    def __iter__(self):
        rp = np.random.permutation(self.pos)
        rn = np.random.permutation(self.neg)
        rh = np.random.permutation(self.hn)
        for i in range(0, min(len(rp),len(rn),len(rh)), self.b):
            yield list(rp[i:i+self.b]) + list(rn[i:i+self.b]) + list(rh[i:i+self.b])
    def __len__(self): return min(len(self.pos),len(self.neg),len(self.hn))//self.b
```

**Temperature scaling**

```python
class TempScale(nn.Module):
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1))
    def forward(self, z): return z / self.T.clamp_min(1e-3)
# fit T on val logits/labels by minimizing CE
```

**FAH hesap/threshold seçimi**

```python
def pick_thr(logits, labels, total_sec, target_fah):
    p = torch.sigmoid(torch.tensor(logits))
    thrs = torch.linspace(0,1,401)
    best = (0.5, 0.0)
    for t in thrs:
        pred = (p>=t).int()
        FP = ((pred==1)&(labels==0)).sum().item()
        TP = ((pred==1)&(labels==1)).sum().item()
        P  = (labels==1).sum().item()
        fah = FP / (total_sec/3600.0 + 1e-9)
        tpr = TP / max(P,1)
        if fah <= target_fah and tpr > best[1]:
            best = (float(t), tpr)
    return best[0]
```

**Streaming dedektör**

```python
class Detector:
    def __init__(self, thr, lockout_ms=1500, vote=3, win=5, hyster=0.1):
        self.thr_on, self.thr_off = thr, max(thr-hyster,0)
        self.lockout_ms, self.vote, self.win = lockout_ms, vote, win
        self.buf, self.locked_until = [], 0
    def step(self, score, now_ms):
        self.buf.append(score); self.buf = self.buf[-self.win:]
        if now_ms < self.locked_until: return False
        hit = sum(s>=self.thr_on for s in self.buf) >= self.vote
        if hit:
            self.locked_until = now_ms + self.lockout_ms
            return True
        return False
```

**Öncelik sırası**

1. Operating-point + FAH ölçümü ve threshold seçimi
2. CMVN + leakage guard + balanced batch
3. EMA + temperature scaling
4. Streaming dedektör + latency ölçümü
5. K-fold ve domain-shift suite
6. ONNX/INT8 export

Feature Cache için kodun durumuna göre aşağıdaki değerlerden birini seç ve feature cache max ram gb belirle. 

#### Feature Cache
8 GB VRAM + 64 GB RAM için **FeatureCache(max_ram_gb=12–16)** ile başla.

ELI5:

* LRU cache = “en son kullanılanlar RAM’de kalsın, eskiler atılsın”.
* `.npy` log-mel (64×T, T≈150 @1.5 s, fp16) ≈ **~20–25 KB/örnek**. fp32 ise ≈ **~40–50 KB**.
* 125k örnek fp16 tümü ≈ **~2.5–3 GB**; fp32 tümü ≈ **~5–6 GB**. 64 GB RAM’in varken tümünü de sığdırırsın, ama OS + DataLoader + diğer süreçler için pay bırak.

Öneri (pratik):

* Başlangıç: `FeatureCache(max_ram_gb=12)`
* GPU util < 80% ve IO beklemesi görürsen: **16** yap.
* fp32 özellik kullanıyorsan: **16–24** yap.
* Her durumda `np.load(..., mmap_mode='r')` açık kalsın; cache “sıcak” alt kümeyi RAM’de tutar.
* `prefetch_factor=4`, `num_workers=16`, `pin_memory=True`, `persistent_workers=True` ile birlikte kullan.

Opsiyonel otomatik ölçek:

* Ortalama özellik dosya boyutunu ilk 1–2k örnekten ölç → `max_items = floor((max_ram_gb*1024**3)/avg_bytes)`; cache bunu sınır olarak kullansın.

Kısaca ayar satırı:

```
FeatureCache(max_ram_gb=16)   # fp16
FeatureCache(max_ram_gb=24)   # fp32 veya uzun pencere
```
