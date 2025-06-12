backend run : uvicorn main:app --reload
frontend run : npm start
backend için gerekli olan model ve label encoder: https://drive.google.com/drive/folders/12NLA-PJjFdNxANZwKoz0QaW_3P2Pz50r?usp=sharing

📊 Kategori Bazlı Metrikler:

  dunya:
    Precision: 0.8901
    Recall: 0.8062
    F1-Score: 0.8461
    Destek (Support): 1125.0

  ekonomi:
    Precision: 0.8960
    Recall: 0.9040
    F1-Score: 0.9000
    Destek (Support): 1125.0

  magazin:
    Precision: 0.9753
    Recall: 0.9831
    F1-Score: 0.9792
    Destek (Support): 1126.0

  saglik:
    Precision: 0.9309
    Recall: 0.9565
    F1-Score: 0.9435
    Destek (Support): 1126.0

  spor:
    Precision: 0.9794
    Recall: 0.9725
    F1-Score: 0.9759
    Destek (Support): 1126.0

  teknoloji:
    Precision: 0.8545
    Recall: 0.9031
    F1-Score: 0.8781
    Destek (Support): 1125.0

    🧪 Final Model Değerlendirmesi
  Accuracy: 0.9209
  Precision (Weighted): 0.9211
  Recall (Weighted): 0.9209
  F1-Score (Weighted): 0.9205


