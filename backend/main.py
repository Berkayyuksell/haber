from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification # BURADA DEĞİŞİKLİK YAPILDI!
import pickle
import numpy as np
from typing import List, Optional # Optional eklendi

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm origin'lere izin ver
    allow_credentials=True,
    allow_methods=["*"], # Tüm HTTP metotlarına (GET, POST vb.) izin ver
    allow_headers=["*"], # Tüm başlıklara izin ver
)

model_dir = "../model" 
label_encoder_path = "../label_encoder.pkl" 

# Modeli ve diğer bileşenleri başlatmak için boş değişkenler
tokenizer: Optional[BertTokenizer] = None
model: Optional[BertForSequenceClassification] = None
label_encoder: Optional[pickle._Unpickler] = None
device: torch.device = torch.device("cpu") # 

@app.on_event("startup")
async def load_resources():
    
    global tokenizer, model, label_encoder, device

    try:
        # Cihazı belirle
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"🚀 Kullanılacak cihaz: {device}")

        
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval() 

        # Label encoder'ı yükle
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        print(f"✅ Model, tokenizer ve label encoder başarıyla yüklendi. Cihaz: {device}")

    except Exception as e:
        print(f"❌ Model veya diğer kaynaklar yüklenirken hata oluştu: {e}")
        print(f"Lütfen '{model_dir}' klasörünün ve '{label_encoder_path}' dosyasının doğru yolda olduğundan ve eğitimden sonra oluşturulduğundan emin olun.")
    


class NewsInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float 

# --- Tahmin Fonksiyonu ---
def predict_category(text: str) -> PredictionResponse: 
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model henüz yüklenmedi veya yüklenirken hata oluştu.")
    max_len_for_inference = 128 
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len_for_inference,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    # Model çıktısını al
    with torch.no_grad(): 
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    # Logit'leri al ve olasılıklara dönüştür (softmax kullanarak)
    # BertForSequenceClassification doğrudan logit'leri döndürür.
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    # En yüksek olasılığa sahip sınıfı ve güven skorunu bul
    confidence, predicted_idx = torch.max(probabilities, 1)
    # Sayısal etiketi kategori adına geri dönüştür
    category = label_encoder.inverse_transform([predicted_idx.item()])[0]
    # Sonuçları PredictionResponse modeline uygun şekilde döndür
    return PredictionResponse(category=category, confidence=confidence.item())

# --- API Endpoints ---

@app.get("/")
async def root():
    """API'nin durumunu kontrol etmek için basit bir endpoint."""
    return {"message": "News Category Classification API is running. Send POST requests to /predict or /predict_multiple."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_news_category(news: NewsInput):
    """Tek bir haber başlığının kategorisini tahmin eder."""
    try:
        result = predict_category(news.text)
        return result
    except HTTPException as e: # HTTPException'ı yakala ve yeniden fırlat
        raise e
    except Exception as e: # Diğer genel hataları yakala
        raise HTTPException(status_code=500, detail=f"Tahmin sırasında beklenmeyen bir hata oluştu: {str(e)}")

@app.post("/predict_multiple", response_model=List[PredictionResponse])
async def predict_multiple_news_categories(news_items: List[NewsInput]):
    """Birden fazla haber başlığının kategorilerini tahmin eder."""
    results = []
    try:
        for news_item in news_items:
            prediction = predict_category(news_item.text)
            results.append(prediction)
        return results
    except HTTPException as e: # HTTPException'ı yakala ve yeniden fırlat
        raise e
    except Exception as e: # Diğer genel hataları yakala
        raise HTTPException(status_code=500, detail=f"Çoklu tahmin sırasında beklenmeyen bir hata oluştu: {str(e)}")