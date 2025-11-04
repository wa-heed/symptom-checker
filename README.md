# Symptom Checker

## Summary
An AI tool that helps users identify possible health conditions based on reported symptoms. The system provides guidance for next steps — such as consulting a doctor — and promotes awareness of common illnesses.

## Background
- **Problem:** People often struggle to determine which illnesses their symptoms indicate, leading to delayed medical attention or unnecessary worry.
- **Frequency:** Millions of people search online daily for symptom guidance, often finding unreliable information.
- **Personal motivation:** I want to help people make informed health decisions and reduce unnecessary stress.
- **Importance:** Early guidance can encourage timely medical consultation, improve health outcomes, and educate users about common conditions.

## How is it used?
- **Context:** Accessible via a simple web or mobile app interface.
- **Users:** General public, especially those experiencing mild or uncertain symptoms.
- **Process:**
  1. User enters their symptoms (e.g., "fever", "headache", "cough").
  2. The AI analyzes symptoms using a trained classification model.
  3. The system suggests possible conditions and gives advice (e.g., "Consult a doctor if symptoms persist").

![Symptom Checker](https://upload.wikimedia.org/wikipedia/commons/4/47/Medical_symptoms_chart.jpg)

## Data sources and AI methods
- **Data:**
  - Open-source datasets such as MedNLI or SymCAT for symptom-to-condition mapping.
  - Optionally, a manually curated dataset of common local illnesses.

- **AI techniques:**
  - Natural Language Processing (NLP) to interpret text-based symptom input.
  - Classification model to predict possible conditions from symptom patterns.
  - Implemented using Python, scikit-learn, or TensorFlow.

## Example Demo Code

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Training data
symptoms = ["fever cough", "headache nausea", "rash itch"]
conditions = ["Flu", "Migraine", "Allergy"]

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)
y = conditions

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Test example
test_symptom = ["cough fever"]
test_vec = vectorizer.transform(test_symptom)
print("Predicted condition:", model.predict(test_vec)[0])
