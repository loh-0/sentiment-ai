# Mental Health Severity Classifier

This is a simple AI-powered triage demo that classifies emotional severity from user messages using a transformer model. It’s designed as a proof of concept for how sentiment analysis can support mental health professionals in prioritising responses — especially in identifying cases at risk of self-harm.

---

## 🔍 What It Does

- Uses a Hugging Face transformer model to classify emotions in text  
- Flags potential suicide-risk messages based on keywords  
- Rates the severity of emotional distress into 4 levels:
  - 🔴 Immediate: Self Harm Mentioned  
  - 🔴 Very Severe  
  - 🟠 Severe  
  - 🟡 Moderate  
  - 🟢 Mild  

---

## 🧠 Model Used

- **Model:** [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)  
- **Pipeline:** `text-classification` from `transformers`

---

## 📦 Installation

### Downlaod Requirements and run demo.py

pip install -r requirements.txt
