##Distilbert from huggingface used on the sentiment prediction for test set to know test accuracy 
from transformers import pipeline
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Path to the text file
file_path = "processed_sentiments_test.txt"

# Load sentences and labels
sentences = []
true_labels = []

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        label, sentence = line.strip().split("\t")  # Assuming tab-separated sentence and label
        sentences.append(sentence)
        # Map neutral to either positive or negative during predictions
        if label.lower() == "neutral":
            true_labels.append(None)  # Neutral labels will not directly impact accuracy
        else:
            true_labels.append(label.lower())  # Convert other labels to lowercase for consistency

# Perform sentiment analysis
predicted_labels = []

for sentence in sentences:
    result = sentiment_analyzer(sentence)[0]  # Get the first result
    predicted_label = result["label"].lower()  # Convert to lowercase for comparison
    predicted_labels.append(predicted_label)

# If true_labels has None (for NEUTRAL), exclude them from accuracy calculation
filtered_true_labels = [label for label in true_labels if label is not None]
filtered_predicted_labels = [pred for label, pred in zip(true_labels, predicted_labels) if label is not None]

# Calculate accuracy
accuracy = accuracy_score(filtered_true_labels, filtered_predicted_labels)

print(f"Accuracy (excluding NEUTRAL): {accuracy:.4f}")

