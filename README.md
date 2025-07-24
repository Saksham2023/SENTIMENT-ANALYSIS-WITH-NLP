# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: SAKSHAM SHRIVASTAVA

INTERN ID: CT04DZ534

DOMAIN: MACHINE LEARNING 

DURATION: 4 WEEKS

MENTOR: NEELA SANTHOSH

Project Description: SENTIMENT
ANALYSIS WITH
NLP

This project combines the power of computer vision and natural language processing (NLP) to perform sentiment analysis on textual data extracted from images. The primary objective is to recognize text from an image using Optical Character Recognition (OCR), clean and transform that text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency), and finally classify the sentiment (positive or negative) using a Logistic Regression model.

The project starts by accepting an image input containing user reviews or opinionated text. Since machine learning models cannot directly understand image data for text-based tasks, we use the Tesseract OCR engine via the pytesseract Python library to extract raw text from the image. Tesseract is a powerful open-source tool that recognizes characters in an image and converts them into machine-readable strings.

Once the text is extracted, the next phase involves text preprocessing. This includes converting the text to lowercase, removing punctuation, eliminating numbers, stripping white spaces, and removing URLs, special characters, or stop words. Cleaning is a crucial step to ensure that the dataset is consistent, noise-free, and ready for vectorization.

The cleaned data is then converted into numerical form using the TF-IDF Vectorizer. TF-IDF assigns importance to each word based on how often it appears in a document versus how frequently it appears across all documents. This helps in giving more weight to meaningful words that are more relevant to a specific review and less weight to common words.

After feature extraction, the next stage is model training. Here, we use Logistic Regression, a linear model commonly used for binary classification problems such as sentiment analysis. Logistic Regression computes the probability of a sample belonging to a particular class (positive or negative sentiment in this case) and makes predictions based on learned weights.

The dataset is split into training and testing sets, usually in an 80:20 ratio. The model is trained on the training data and then evaluated on the testing set to measure performance. Evaluation metrics like precision, recall, F1-score, and accuracy are used to determine how well the model is classifying sentiments.

To visually assess the model’s performance, a confusion matrix is plotted using seaborn. This matrix shows the number of true positives, true negatives, false positives, and false negatives, helping to understand where the model performs well and where it makes mistakes.

This end-to-end pipeline demonstrates a practical application of combining image processing, OCR, text cleaning, feature engineering, and machine learning to build an automated sentiment analysis system. Such systems can be used in real-world applications like social media monitoring, customer review analysis, brand sentiment tracking, or feedback evaluation — even when the data is received in the form of images (like scanned reviews or handwritten notes).

In conclusion, this project showcases how integrating multiple domains of AI — computer vision and NLP — can lead to robust and intelligent systems capable of analyzing unstructured image-based textual data and deriving valuable insights from it.

OUTPUT OF MY CODE 
<img width="617" height="586" alt="image" src="https://github.com/user-attachments/assets/49da73e6-95d1-4762-abcc-a7c92184745d" />

