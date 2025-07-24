# -------------------------------
# Sentiment Analysis from Image
# -------------------------------

# 📦 Step 1: Import Libraries
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import cv2
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 📌 Step 2: Set Tesseract Path (Update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🖼️ Step 3: Read and OCR the Image
image_path = r"C:\Users\HP\Downloads\WhatsApp Image 2025-07-16 at 2.24.00 PM.jpeg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 🧠 Step 4: Extract Text
extracted_text = pytesseract.image_to_string(image_rgb)
print("🔹 Extracted Text:\n")
print(extracted_text)

# 🧪 Step 5: Create a Sample Review Dataset (you can change this later)
data = {
    'review': [
        "The product was excellent and delivery was fast.",
        "Worst purchase ever. Completely useless.",
        "I loved the customer service and packaging.",
        "Very bad experience. Not recommended at all.",
        "Amazing quality, very satisfied.",
        "Terrible item, waste of money.",
    ],
    'sentiment': [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# 🧼 Step 6: Clean the Reviews
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# 🔢 Step 7: Split Dataset
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧮 Step 8: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 🧠 Step 9: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 📈 Step 10: Predict & Evaluate
y_pred = model.predict(X_test_tfidf)

print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred))

# 📊 Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
