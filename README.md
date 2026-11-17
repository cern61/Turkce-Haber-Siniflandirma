# 📰 Türkçe Haber Sınıflandırma

**BİL482 Doğal Dil İşleme — Uygulama Ödevi**

Bu çalışma, Türkçe haber metinlerini içeriklerine göre **yedi konu kategorisinden birine** atayan uçtan uca bir doğal dil işleme (NLP) uygulamasıdır. Veri hazırlama ve Türkçe metin ön işlemeden başlayarak farklı metin temsilleri ve sınıflandırma modelleri karşılaştırılır.

> 📓 Uygulamanın kodu, açıklamaları ve görselleştirmeleri [`Turkce_Haber_Siniflandirma.ipynb`](./Turkce_Haber_Siniflandirma.ipynb) defterindedir.

## 🎯 Ödevin amacı

Haber metinlerinden anlamlı özellikler çıkarıp modellerin sınıflandırma başarısını karşılaştırmak. Değerlendirmede **macro-F1** temel ölçüt olarak kullanılır; accuracy ve sınıf bazlı ölçümler de incelenir.

## 🗂️ Veri seti

Projede, Türkçe haberlerden oluşan **TTC4900** veri seti kullanılır. Veri; `metin` (haber içeriği) ve `kategori` (hedef etiket) alanlarıyla çalışılır.

| Kategori | Kategori | Kategori |
|---|---|---|
| Dünya (`dunya`) | Ekonomi (`ekonomi`) | Kültür (`kultur`) |
| Sağlık (`saglik`) | Siyaset (`siyaset`) | Spor (`spor`) |
| Teknoloji (`teknoloji`) |  |  |

Veri, Hugging Face veri seti üzerinden yüklenir; gerektiğinde CSV bağlantısı yedek kaynak olarak kullanılır. Veri setine erişim için internet bağlantısı gerekir.

- [TTC4900 — Hugging Face](https://huggingface.co/datasets/savasy/ttc4900)
- [Kaynak CSV](https://raw.githubusercontent.com/savasy/TurkishTextClassification/master/7allV03.csv)

## 🔬 Uygulama akışı

Defter, adımları açıklamalar ve ara çıktılar eşliğinde takip eder:

1. **Veriyi yükleme ve inceleme:** Veri seti alınır, sütunlar düzenlenir; sınıf dağılımı ve metinler gözden geçirilir.
2. **Türkçe metin ön işleme:** Harf dönüşümü Türkçeye uygun biçimde ele alınır; URL, e-posta, sayı ve noktalama gibi içerikler temizlenir. Türkçe harfler korunur.
3. **Dilbilgisel çözümleme:** NLTK ve Stanza kullanılarak metin işleme ve sözcük türü (UPOS) incelemeleri yapılır.
4. **Metin temsilleri:** Sözcük torbası (BoW), TF-IDF, Word2Vec ve BERTurk temsilleri oluşturulur.
5. **Modelleme ve değerlendirme:** Naive Bayes, Lojistik Regresyon, Doğrusal SVM ve Random Forest gibi sınıflandırıcılar karşılaştırılır. Sonuçlar tablolar ve grafiklerle sunulur.
6. **Temsil karşılaştırması:** TF-IDF ve BERTurk yaklaşımları aynı veri bölünmesi üzerinde ayrıca değerlendirilir.

## 🚀 Çalıştırma

En kolay yol **Google Colab** veya Jupyter Notebook kullanmaktır:

1. Depoyu bilgisayarınıza indirin ya da klonlayın.
2. [`Turkce_Haber_Siniflandirma.ipynb`](./Turkce_Haber_Siniflandirma.ipynb) dosyasını Colab veya Jupyter’da açın.
3. Hücreleri sırasıyla çalıştırın. Kurulum hücresi gerekli Python paketlerini ve NLTK/Stanza kaynaklarını indirir.

> ⚙️ İlk çalıştırmada paketlerin, dil kaynaklarının ve BERTurk modelinin indirilmesi zaman alabilir. İnternet bağlantısı ve yeterli çalışma belleği gereklidir. Büyük model adımları için Google Colab’ın GPU çalışma zamanı tercih edilebilir.

Notebook, kaynakların kullanılabilirliğine ve çalışma ortamına bağlı olarak bazı aşamalarda örneklem kullanabilir. Bu nedenle çalışma süresi ve elde edilen metrikler çalıştırma koşullarına göre değişebilir; güncel sonuçlar notebook çıktılarında yer alır.

## 🧰 Kullanılan araçlar

**Python · pandas · NumPy · scikit-learn · NLTK · Stanza · PyTorch · Transformers · Matplotlib · Seaborn**

## 📁 Depo içeriği

```text
.
├── README.md
└── Turkce_Haber_Siniflandirma.ipynb
```

## 📚 Kaynaklar

- [TTC4900 veri seti](https://huggingface.co/datasets/savasy/ttc4900)
- [TurkishTextClassification veri seti kaynağı](https://github.com/savasy/TurkishTextClassification)
- [BERTurk — dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)