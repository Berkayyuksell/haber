import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm # Eğitim döngüsünde ilerleme çubuğu için

# --- BERT Model Part ---

# GPU/MPS/CPU kullanımını kontrol eden fonksiyon
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU bulundu! {gpu_count} adet kullanılabilir GPU.")
        print(f"🔥 Kullanılan GPU: {gpu_name}")
        if hasattr(torch.cuda, 'get_device_properties'):
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"📊 GPU bellek: {total_memory:.2f} GB")
    elif torch.backends.mps.is_available(): # Apple Silicon (M1/M2) için MPS kontrolü
        device = torch.device("mps")
        print("🍏 Apple Silicon (MPS) bulundu! GPU hızlandırması kullanılacak.")
        # MPS için bellek bilgisi doğrudan alınamayabilir, sistem bilgisi daha geneldir.
    else:
        device = torch.device("cpu")
        print("⚠️ GPU veya MPS bulunamadı, CPU kullanılacak (Bu işlem çok yavaş olabilir!)")
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"💻 CPU çekirdek sayısı: {cpu_count}")
    
    return device

class NewsDataset(Dataset):
    """BERT modeli için veri seti sınıfı."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data_from_csv(file_path='total_cleaned_lowercase.csv'):
    """CSV dosyasından veri yükler."""
    try:
        df = pd.read_csv(file_path)
        print(f"CSV dosyasından {len(df)} kayıt başarıyla yüklendi.")
        
        # Sütun isimlerini kontrol et
        if 'text' not in df.columns or 'category' not in df.columns:
            # En iyi tahmin: İlk iki sütun text ve category olabilir
            if len(df.columns) >= 2:
                # Kullanıcıdan onay almak daha iyi olabilir veya belirli bir varsayımda bulunmak
                print("⚠️ 'text' veya 'category' sütunları bulunamadı. İlk sütunu 'text', ikinci sütunu 'category' olarak kabul ediliyor.")
                df = df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'category'})
            else:
                raise ValueError("CSV dosyasında 'text' ve 'category' sütunları veya yeterli sütun bulunmuyor.")
        
        # NaN değerleri kontrol et ve düşür
        initial_rows = len(df)
        df.dropna(subset=['text', 'category'], inplace=True)
        if len(df) < initial_rows:
            print(f"🧹 NaN değerler içeren {initial_rows - len(df)} satır düşürüldü.")

        # Kategori değerlerini kontrol et
        print(f"Kategoriler: {df['category'].unique()}")
        print(f"Kategori dağılımı:\n{df['category'].value_counts()}")
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Hata: '{file_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return None
    except pd.errors.EmptyDataError:
        print(f"❌ Hata: '{file_path}' dosyası boş.")
        return None
    except Exception as e:
        print(f"❌ CSV dosyası yüklenirken beklenmeyen bir hata oluştu: {str(e)}")
        return None

def balance_dataset(df, samples_per_category=None):
    """Veri setini her kategoriden belirli sayıda örnek içerecek şekilde dengele.
    samples_per_category None ise, en az örneğe sahip kategorinin sayısına göre dengeleme yapılır.
    """
    balanced_df = pd.DataFrame()
    
    if samples_per_category is None:
        min_samples = df['category'].value_counts().min()
        samples_per_category = min_samples
        print(f"ℹ️ `samples_per_category` belirtilmedi. En az örneğe sahip kategori ({df['category'].value_counts().idxmax()}) sayısı ({min_samples}) kullanılacak.")
        
    print(f"📊 Her kategori için hedeflenen örnek sayısı: {samples_per_category}")

    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        if len(category_df) > samples_per_category:
            # Rastgele örnekleme
            category_df = category_df.sample(n=samples_per_category, random_state=42)
        elif len(category_df) < samples_per_category:
            print(f"⚠️ Kategori '{category}' için yeterli örnek yok ({len(category_df)}/{samples_per_category}). Mevcut tüm örnekler kullanılacak.")
        balanced_df = pd.concat([balanced_df, category_df])
    
    # Karışık hale getir
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"📊 Dengelenmiş veri seti boyutu: {len(balanced_df)} örnek")
    print(f"📊 Kategori dağılımı (Dengeleme sonrası):\n{balanced_df['category'].value_counts()}")
    
    return balanced_df

def evaluate_model(model, data_loader, device, label_encoder, plot_confusion_matrix=False):
    """Model performansını değerlendir ve metrikleri hesapla."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Değerlendirme"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, predicted = torch.max(outputs.logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrikleri hesapla
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Sınıflandırma raporu
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=label_encoder.classes_, 
        output_dict=True, 
        zero_division=0 # Desteklenmeyen sınıflar için 0 atar
    )
    
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    # Kategori bazlı metrikler
    category_metrics = {
        category: {
            'precision': report[category]['precision'],
            'recall': report[category]['recall'],
            'f1': report[category]['f1-score'],
            'support': report[category]['support']
        } for category in label_encoder.classes_
    }

    if plot_confusion_matrix:
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title('Karmaşıklık Matrisi')
        plt.xlabel('Tahmin Edilen Etiket')
        plt.ylabel('Gerçek Etiket')
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'category_metrics': category_metrics,
        'all_predictions': all_predictions,
        'all_labels': all_labels # Karmaşıklık matrisi için
    }

def train_model(df):
    text_col = 'text'
    label_col = 'category'
    
    df = balance_dataset(df, samples_per_category=None) 
    
    # Label encoding
    le = LabelEncoder()
    df['encoded_category'] = le.fit_transform(df[label_col])
    num_labels = len(le.classes_)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].values, 
        df['encoded_category'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['encoded_category'] # Kategorilere göre stratify et
    )
    
    # Tokenizer yükle
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    
    # Cihaz belirlemesi ve batch boyutu
    device = get_device()
    # M2 MacBook Air için MPS kullanılıyorsa daha büyük batch boyutu deneyebiliriz
    batch_size = 32 if str(device) == 'mps' else (16 if str(device) == 'cuda' else 8)
    # Metin uzunluğunu göz önünde bulundurarak max_len ayarı
    max_len = 128 # Haber başlıkları için yeterli, daha uzun metinler için 256 veya 512
    
    print(f"✅ Batch boyutu: {batch_size}")
    print(f"✅ Max token uzunluğu: {max_len}")
    print(f"📊 Eğitim seti boyutu: {len(X_train)} örnek")
    print(f"📊 Test seti boyutu: {len(X_test)} örnek")
    
    # Veri setlerini oluştur
    train_dataset = NewsDataset(X_train, y_train, tokenizer, max_len=max_len)
    test_dataset = NewsDataset(X_test, y_test, tokenizer, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2 if str(device) == 'cpu' else 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count()//2 if str(device) == 'cpu' else 0)
    
    # Model
    model = BertForSequenceClassification.from_pretrained(
        'dbmdz/bert-base-turkish-cased',
        num_labels=num_labels
    )
    
    # Cihaza taşı
    model.to(device)
    print(f"ℹ️ Model {device} üzerinde eğitiliyor...")
    
    # Optimizer ve Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5) 
    epochs = 3 
    total_steps = len(train_loader) * epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # Isınma adımı yok
        num_training_steps=total_steps
    )
    
    print(f"🔄 Eğitim başlıyor: {epochs} epoch, toplam {total_steps} batch")
    
    best_f1 = -1 # F1 skoru 0 ile 1 arasında olduğu için -1 ile başlat
    best_model_state = None
    
    # Eğitim geçmişi
    train_losses = []
    eval_f1_scores = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Eğitimi"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradyan patlamasını önlemek için
            optimizer.step()
            scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'  Epoch {epoch+1}/{epochs} tamamlandı, Ortalama loss: {avg_loss:.4f}')
        
        # Her epoch sonunda modeli değerlendir
        metrics = evaluate_model(model, test_loader, device, le, plot_confusion_matrix=False) # Her epochta çizimi kapat
        eval_f1_scores.append(metrics['f1'])
        print("\n📊 Epoch Değerlendirme Metrikleri:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        
        # En iyi modeli kaydet (F1 skoru bazında)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            # Modelin durumunu kopyala
            best_model_state = model.state_dict()
            print(f"  🏆 Yeni en iyi model kaydedildi! (F1: {best_f1:.4f})")
    
    # En iyi modeli yükle (eğer bir iyileşme olduysa)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n✅ En iyi model durumu yüklendi.")
    else:
        print("\nℹ️ Modelde herhangi bir F1 skoru iyileşmesi görülmedi. Son epoch modeli kullanılacak.")
    
    # Eğitim geçmişini çiz
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, marker='o')
    plt.title('Eğitim Kaybı (Training Loss) vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), eval_f1_scores, marker='o', color='green')
    plt.title('F1-Score (Test Seti) vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Final değerlendirme (karmaşıklık matrisi ile)
    print("\n🧪 Final Model Değerlendirmesi:")
    final_metrics = evaluate_model(model, test_loader, device, le, plot_confusion_matrix=True) # Sonuçları görselleştir
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Precision (Weighted): {final_metrics['precision']:.4f}")
    print(f"  Recall (Weighted): {final_metrics['recall']:.4f}")
    print(f"  F1-Score (Weighted): {final_metrics['f1']:.4f}")
    
    print("\n📊 Kategori Bazlı Metrikler:")
    for category, metrics in final_metrics['category_metrics'].items():
        print(f"\n  {category}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1']:.4f}")
        print(f"    Destek (Support): {metrics['support']}")
    
    # Modeli kaydet
    model_save_path = 'news_category_model'
    os.makedirs(model_save_path, exist_ok=True) # Klasör yoksa oluştur
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Label encoder'ı kaydet
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\n💾 Model ve tokenizer kaydedildi: {model_save_path}")
    print(f"💾 Label encoder kaydedildi: label_encoder.pkl")
    
    return model, tokenizer, le

def predict_category(headline, model, tokenizer, label_encoder, device=None):
    """Verilen haber başlığının kategorisini tahmin eder."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    encoding = tokenizer.encode_plus(
        headline,
        add_special_tokens=True,
        max_length=128, # Eğitimde kullanılan max_len ile uyumlu olmalı
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    predicted_category = label_encoder.inverse_transform([predicted.item()])[0]
    
    return predicted_category, confidence.item()

# --- Chatbot Class ---

class NewsCategoryBot:
    """Haber kategorisi tahmin eden chatbot sınıfı."""
    def __init__(self):
        self.device = get_device() # Cihazı başlatmada belirle
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.categories = []
        
        self._load_model_and_tokenizer() # Başlangıçta modeli yüklemeyi dene
    
    def _load_model_and_tokenizer(self):
        """Kaydedilmiş modeli ve tokenizer'ı yüklemeye çalışır."""
        model_path = 'news_category_model'
        label_encoder_path = 'label_encoder.pkl'
        
        if os.path.exists(model_path) and os.path.exists(label_encoder_path):
            try:
                print(f"💾 Kaydedilmiş model ve tokenizer bulundu: {model_path}")
                self.model = BertForSequenceClassification.from_pretrained(model_path)
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                self.categories = list(self.label_encoder.classes_)
                print(f"📋 Model {len(self.categories)} farklı kategori tahmin edebilir:")
                print(f"   {', '.join(self.categories)}")
                
                print("✅ Model başarıyla yüklendi!")
                self.model_loaded = True
                self.model.to(self.device) # Yüklenen modeli cihaza taşı
            except Exception as e:
                print(f"⚠️ Kaydedilmiş model yüklenirken hata oluştu: {str(e)}")
                print("🔄 Yeni model eğitilmesi gerekebilir.")
                self.model_loaded = False
        else:
            print("⚠️ Kaydedilmiş model veya label encoder bulunamadı.")
            print("🔄 Yeni model eğitilmesi gerekiyor.")
            self.model_loaded = False
    
    def train(self):
        """Modeli eğit."""
        if self.model_loaded:
            print("ℹ️ Model zaten yüklü ve eğitilmiş durumda. Yeniden eğitmeye devam ediliyor...")
            confirm = input("Eğitim verileri sıfırlanacak ve model yeniden eğitilecek. Emin misiniz? (evet/hayır): ").lower()
            if confirm != 'evet':
                print("Eğitim iptal edildi.")
                return False

        print("📊 Veri setini yükleniyor...")
        df = load_data_from_csv()
        
        if df is not None and not df.empty:
            print("🧠 Model eğitimi başlıyor...")
            self.model, self.tokenizer, self.label_encoder = train_model(df)
            self.categories = list(self.label_encoder.classes_)
            self.model_loaded = True
            return True
        else:
            print("❌ CSV dosyasından geçerli veri yüklenemedi veya veri boş. Model eğitilemedi.")
            return False
    
    def predict(self, headline):
        """Haber başlığının kategorisini tahmin et."""
        if not self.model_loaded:
            return "Model henüz eğitilmedi. Lütfen önce 'train' fonksiyonunu çağırın.", 0.0
        
        category, confidence = predict_category(headline, self.model, self.tokenizer, self.label_encoder, self.device)
        return category, confidence
    
    def respond(self, user_input):
        """Kullanıcı girdisine yanıt verir."""
        user_input_lower = user_input.lower().strip()

        if user_input_lower in ["eğit", "train", "egit"]:
            success = self.train()
            if success:
                return "✅ Model eğitildi ve kaydedildi."
            else:
                return "❌ Model eğitimi başarısız oldu!"
        elif user_input_lower in ["kategoriler", "categories"]:
            if self.model_loaded and self.categories:
                categories_str = ", ".join(self.categories)
                return f"📋 Mevcut kategoriler: {categories_str}"
            else:
                return "❌ Model henüz eğitilmedi veya kategori bilgisi yok."
        else:
            category, confidence = self.predict(user_input)
            confidence_percent = confidence * 100
            
            certainty_level = ""
            if confidence_percent >= 95:
                certainty_level = "kesinlikle"
            elif confidence_percent >= 80:
                certainty_level = "çok yüksek olasılıkla"
            elif confidence_percent >= 60:
                certainty_level = "yüksek olasılıkla"
            elif confidence_percent >= 40:
                certainty_level = "orta olasılıkla"
            else:
                certainty_level = "düşük olasılıkla"
                
            return f'📰 Tahmin edilen kategori: **{category}** (güven: %{confidence_percent:.2f} - {certainty_level})'

# --- Main Function ---

def main():
    """Ana program fonksiyonu."""
    print("""
    📰 HABER KATEGORİ TAHMİN BOTU 📰
    ===============================
    
    Kullanım:
    - Bir haber başlığı yazın, bot kategorisini tahmin edecektir.
    - "train" veya "eğit" yazarak modeli yeniden eğitebilirsiniz.
    - "kategoriler" yazarak mevcut kategorileri görebilirsiniz.
    - "çıkış" yazarak programdan çıkabilirsiniz.
    """)
    
    bot = NewsCategoryBot()
    
    # Model ilk başta yüklenemezse, kullanıcıdan eğitmesini iste
    if not bot.model_loaded:
        print("\nModel hazır değil. Lütfen önce modeli eğitin veya bir başlık girin.")
        print("Modeli eğitmek için 'train' yazın.")
        
    # Chat döngüsü
    while True:
        user_input = input("\n📌 Haber başlığı girin (çıkış için 'çıkış' yazın): ")
        
        if user_input.lower().strip() == "çıkış":
            print("👋 Program sonlandırılıyor...")
            break
        
        response = bot.respond(user_input)
        print(response)

if __name__ == "__main__":
    main()