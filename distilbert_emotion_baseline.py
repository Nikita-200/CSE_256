#Distilbert from huggingface used on the emotion prediction for test set to know test accuracy 
from transformers import pipeline

# Load the emotion classification pipeline
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Initialize variables for accuracy calculation
total_samples = 0
correct_predictions = 0
sentences = []  # To store the text sentences
true_labels = []  # To store the true emotions
predicted_labels = []  # To store the predicted emotions

# Open and read the tests_emotion.txt file
with open("tests_emotion.txt", "r", encoding="utf-8") as file:
    for line in file:
        # Split each line into target emotion and text
        target_emotion, text = line.strip().split("\t")
        
        # Store the sentence and true label
        sentences.append(text)
        true_labels.append(target_emotion)
        
        # Use the pipeline to predict emotion
        result = emotion_pipeline(text)
        predicted_emotion = result[0]['label'].lower()
        
        # Store the predicted label
        predicted_labels.append(predicted_emotion)
        
        # Check if the predicted emotion matches the target emotion
        if predicted_emotion.lower() == target_emotion.lower():
            correct_predictions += 1
        total_samples += 1

# Calculate and print the accuracy
accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")
