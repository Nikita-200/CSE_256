import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gmf_model import GraphMemoryFusionNetwork
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_and_evaluate_dfg_model(train_loader, test_loader, input_dims, output_dim=7, epochs=20, learning_rate=1e-3):
    """
    Train and evaluate the DFG model on the given train and test loaders.
    
    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        input_dims (list): List of input dimensions for language, vision, and acoustic features.
        output_dim (int): Number of output classes (default: 7).
        epochs (int): Number of epochs to train for (default: 10).
        learning_rate (float): Learning rate for the optimizer (default: 1e-3).
    """
    # Initialize model, optimizer, and loss function
    print("Initializing the Graph Memory Fusion Network (GMFN)...")
    model = GraphMemoryFusionNetwork(input_dims, output_dim=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion_sentiment = nn.SmoothL1Loss() # For sentiment regression (scaled [-3, 3])
    criterion_emotion = nn.SmoothL1Loss()

    print("Starting training...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        all_train_labels = []
        all_train_preds = []

        for language, vision, acoustic, labels in train_loader:
            language = language.float()  #language.shape torch.Size([32, 50, 300])
            vision = vision.float()  #vision.shape torch.Size([32, 50, 746])
            acoustic = acoustic.float()  #acoustic.shape torch.Size([32, 50, 74])  
            labels = labels.float().squeeze(1)  # Adjust label shape [batch_size, 1, 7] -> [batch_size, 7] 

            torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection globally
            optimizer.zero_grad()

            # Split labels into sentiment and emotions
            sentiment_labels = labels[:, 0].float()  # Sentiment within [-3, 3]
            emotion_labels = labels[:, 1:].float()   # Emotions (6 dimensions) within [0, 3]

            sentiment_preds, emotion_preds = model(language, vision, acoustic)
            
            # Compute losses
            loss_sentiment = criterion_sentiment(sentiment_preds.squeeze(), sentiment_labels)
            loss_emotions = sum(criterion_emotion(emotion_preds[:, i], emotion_labels[:, i]) for i in range(6))
            alpha = 0.6
            loss = alpha * loss_sentiment + (1 - alpha) * loss_emotions  # Experiment with alpha (e.g., 0.6)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)  # Gradient clipping clip_value=5
            optimizer.step()

            train_loss += loss.item()
            sentiment_preds_combine = (sentiment_preds.squeeze()).detach().numpy()
            emotion_preds_combine = (emotion_preds).detach().numpy() 
            combined_preds = np.concatenate([sentiment_preds_combine[:, None], emotion_preds_combine], axis=1)

            all_train_preds.extend(combined_preds)
            all_train_labels.extend(labels.cpu().numpy())

        all_training_labels = np.array(all_train_labels).flatten()
        all_training_preds = np.array(all_train_preds).flatten()

        rounded_labels = np.round(all_training_labels).astype(int)
        rounded_preds = np.round(all_training_preds).astype(int)

        train_acc = accuracy_score(rounded_labels, rounded_preds)
        print(f"Training Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Calculate regression metrics
        mse = mean_squared_error(all_training_labels, all_training_preds)
        mae = mean_absolute_error(all_training_labels, all_training_preds)
        print(f"Training MSE: {mse:.4f}, MAE: {mae:.4f}")

        # Testing phase
        model.eval()
        test_loss = 0.0
        all_test_labels = []
        all_test_preds = []
        test_acc = 0.0

        with torch.no_grad():
            for language, vision, acoustic, labels in test_loader:
                language = language.float()
                vision = vision.float()
                acoustic = acoustic.float()
                labels = labels.float().squeeze(1)  # Adjust label shape [batch_size, 1, 7] -> [batch_size, 7]

                sentiment_labels = labels[:, 0].float()  # Sentiment
                emotion_labels = labels[:, 1:].float()   # Emotions (6 dimensions)
                sentiment_preds, emotion_preds = model(language, vision, acoustic)
                loss_sentiment = criterion_sentiment(sentiment_preds.squeeze(), sentiment_labels)
                loss_emotions = sum(criterion_emotion(emotion_preds[:, i], emotion_labels[:, i]) for i in range(6))
                alpha = 0.6
                loss = alpha * loss_sentiment + (1 - alpha) * loss_emotions  # Experiment with alpha (e.g., 0.6)

                test_loss += loss.item()
                sentiment_preds_combine = (sentiment_preds.squeeze()).detach().numpy()
                emotion_preds_combine = (emotion_preds).detach().numpy()
                combined_preds = np.concatenate([sentiment_preds_combine[:, None], emotion_preds_combine], axis=1)
                all_test_preds.extend(combined_preds)
                all_test_labels.extend(labels.cpu().numpy())

        all_testing_labels = np.array(all_test_labels).flatten()
        all_testing_preds = np.array(all_test_preds).flatten()

        rounded_labels = np.round(all_testing_labels).astype(int)
        rounded_preds = np.round(all_testing_preds).astype(int)

        test_acc = accuracy_score(rounded_labels, rounded_preds)
        print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc:.4f}")

    # Final Metrics
    print("\nFinal Training and Testing Metrics:")

    rounded_train_labels = np.round(all_training_labels).astype(int)
    rounded_train_preds = np.round(all_training_preds).astype(int)
    output_file = "rounded_train_total.txt"

    with open(output_file, "w") as file:
        file.write("Rounded Train Labels:\n")
        file.write(", ".join(map(str, rounded_train_labels)) + "\n\n")
        file.write("Rounded Train Predictions:\n")
        file.write(", ".join(map(str, rounded_train_preds)) + "\n")

    train_accuracy = accuracy_score(rounded_train_labels, rounded_train_preds)
    train_precision = precision_score(rounded_train_labels, rounded_train_preds, average="weighted", zero_division=1)
    train_recall = recall_score(rounded_train_labels, rounded_train_preds, average="weighted", zero_division=1)
    train_f1 = f1_score(rounded_train_labels, rounded_train_preds, average="weighted", zero_division=1)

    # Initialize arrays for each category (7 values per row)
    sentiment_train_labels = np.zeros(len(train_loader), dtype=float)
    happiness_train_labels  = np.zeros(len(train_loader), dtype=float)
    sadness_train_labels  = np.zeros(len(train_loader), dtype=float)
    anger_train_labels  = np.zeros(len(train_loader), dtype=float)
    disgust_train_labels  = np.zeros(len(train_loader), dtype=float)
    surprise_train_labels  = np.zeros(len(train_loader), dtype=float)
    fear_train_labels  = np.zeros(len(train_loader), dtype=float)

    # Process the values in A in chunks of 7 values
    for i in range(len(train_loader)):  # Loop through each "row" of values
        n = i * 7  # Start at the correct index for each row
        if(n+6)>rounded_train_labels.size:
            break
        
        sentiment_train_labels[i] = rounded_train_labels[n]
        happiness_train_labels[i] = rounded_train_labels[n + 1]
        sadness_train_labels[i] = rounded_train_labels[n + 2]
        anger_train_labels[i] = rounded_train_labels[n + 3]
        surprise_train_labels[i] = rounded_train_labels[n + 4]
        disgust_train_labels[i] = rounded_train_labels[n + 5]
        fear_train_labels[i] = rounded_train_labels[n + 6]

    sentiment_train_preds = np.zeros(len(train_loader), dtype=float)
    happiness_train_preds  = np.zeros(len(train_loader), dtype=float)
    sadness_train_preds  = np.zeros(len(train_loader), dtype=float)
    anger_train_preds  = np.zeros(len(train_loader), dtype=float)
    disgust_train_preds  = np.zeros(len(train_loader), dtype=float)
    surprise_train_preds  = np.zeros(len(train_loader), dtype=float)
    fear_train_preds  = np.zeros(len(train_loader), dtype=float)

    for i in range(len(train_loader)):  # Loop through each "row" of values
        n = i * 7  # Start at the correct index for each row
        if(n+6)>rounded_train_preds.size:
            break
        
        sentiment_train_preds[i] = rounded_train_preds[n]
        happiness_train_preds[i] = rounded_train_preds[n + 1]
        sadness_train_preds[i] = rounded_train_preds[n + 2]
        anger_train_preds[i] = rounded_train_preds[n + 3]
        surprise_train_preds[i] = rounded_train_preds[n + 4]
        disgust_train_preds[i] = rounded_train_preds[n + 5]
        fear_train_preds[i] = rounded_train_preds[n + 6]

    sentiment_train_accuracy = accuracy_score(sentiment_train_labels, sentiment_train_preds)
    happiness_train_accuracy = accuracy_score(happiness_train_labels, happiness_train_preds)
    sadness_train_accuracy = accuracy_score(sadness_train_labels, sadness_train_preds)
    anger_train_accuracy = accuracy_score(anger_train_labels, anger_train_preds)
    disgust_train_accuracy = accuracy_score(disgust_train_labels, disgust_train_preds)
    surprise_train_accuracy = accuracy_score(surprise_train_labels, surprise_train_preds)
    fear_train_accuracy = accuracy_score(fear_train_labels, fear_train_preds)

    sentiment_train_precision = precision_score(sentiment_train_labels, sentiment_train_preds, average="weighted", zero_division=1)
    sentiment_train_recall = recall_score(sentiment_train_labels, sentiment_train_preds, average="weighted", zero_division=1)
    sentiment_train_f1 = f1_score(sentiment_train_labels, sentiment_train_preds, average="weighted", zero_division=1)

    happiness_train_precision = precision_score(happiness_train_labels, happiness_train_preds, average="weighted", zero_division=1)
    happiness_train_recall = recall_score(happiness_train_labels, happiness_train_preds, average="weighted", zero_division=1)
    happiness_train_f1 = f1_score(happiness_train_labels, happiness_train_preds, average="weighted", zero_division=1)

    sadness_train_precision = precision_score(sadness_train_labels, sadness_train_preds, average="weighted", zero_division=1)
    sadness_train_recall = recall_score(sadness_train_labels, sadness_train_preds, average="weighted", zero_division=1)
    sadness_train_f1 = f1_score(sadness_train_labels, sadness_train_preds, average="weighted", zero_division=1)

    anger_train_precision = precision_score(anger_train_labels, anger_train_preds, average="weighted", zero_division=1)
    anger_train_recall = recall_score(anger_train_labels, anger_train_preds, average="weighted", zero_division=1)
    anger_train_f1 = f1_score(anger_train_labels, anger_train_preds, average="weighted", zero_division=1)

    disgust_train_precision = precision_score(disgust_train_labels, disgust_train_preds, average="weighted", zero_division=1)
    disgust_train_recall = recall_score(disgust_train_labels, disgust_train_preds, average="weighted", zero_division=1)
    disgust_train_f1 = f1_score(disgust_train_labels, disgust_train_preds, average="weighted", zero_division=1)

    surprise_train_precision = precision_score(surprise_train_labels, surprise_train_preds, average="weighted", zero_division=1)
    surprise_train_recall = recall_score(surprise_train_labels, surprise_train_preds, average="weighted", zero_division=1)
    surprise_train_f1 = f1_score(surprise_train_labels, surprise_train_preds, average="weighted", zero_division=1)

    fear_train_precision = precision_score(fear_train_labels, fear_train_preds, average="weighted", zero_division=1)
    fear_train_recall = recall_score(fear_train_labels, fear_train_preds, average="weighted", zero_division=1)
    fear_train_f1 = f1_score(fear_train_labels, fear_train_preds, average="weighted", zero_division=1)

    print("### Training Metrics ###")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1-Score: {train_f1:.4f}")
    print(f"\nSentiment Train Accuracy: {sentiment_train_accuracy:.4f}")
    print(f"Sentiment Train Precision: {sentiment_train_precision:.4f}")
    print(f"Sentiment Train Recall: {sentiment_train_recall:.4f}")
    print(f"Sentiment Train F1-Score: {sentiment_train_f1:.4f}")
    mae = mean_absolute_error(sentiment_train_labels, sentiment_train_preds)
    print(f"Sentiment Train MAE: {mae:.4f}")

    print(f"\nHappiness Train Accuracy: {happiness_train_accuracy:.4f}")
    print(f"Happiness Train Precision: {happiness_train_precision:.4f}")
    print(f"Happiness Train Recall: {happiness_train_recall:.4f}")
    print(f"Happiness Train F1-Score: {happiness_train_f1:.4f}")

    print(f"\nSadness Train Accuracy: {sadness_train_accuracy:.4f}")
    print(f"Sadness Train Precision: {sadness_train_precision:.4f}")
    print(f"Sadness Train Recall: {sadness_train_recall:.4f}")
    print(f"Sadness Train F1-Score: {sadness_train_f1:.4f}")

    print(f"\nAnger Train Accuracy: {anger_train_accuracy:.4f}")
    print(f"Anger Train Precision: {anger_train_precision:.4f}")
    print(f"Anger Train Recall: {anger_train_recall:.4f}")
    print(f"Anger Train F1-Score: {anger_train_f1:.4f}")

    print(f"\nDisgust Train Accuracy: {disgust_train_accuracy:.4f}")
    print(f"Disgust Train Precision: {disgust_train_precision:.4f}")
    print(f"Disgust Train Recall: {disgust_train_recall:.4f}")
    print(f"Disgust Train F1-Score: {disgust_train_f1:.4f}")

    print(f"\nSurprise Train Accuracy: {surprise_train_accuracy:.4f}")
    print(f"Surprise Train Precision: {surprise_train_precision:.4f}")
    print(f"Surprise Train Recall: {surprise_train_recall:.4f}")
    print(f"Surprise Train F1-Score: {surprise_train_f1:.4f}")

    print(f"\nFear Train Accuracy: {fear_train_accuracy:.4f}")
    print(f"Fear Train Precision: {fear_train_precision:.4f}")
    print(f"Fear Train Recall: {fear_train_recall:.4f}")
    print(f"Fear Train F1-Score: {fear_train_f1:.4f}")

    # Testing Metrics
    rounded_test_labels = np.round(all_testing_labels).astype(int)
    rounded_test_preds = np.round(all_testing_preds).astype(int)
    output_file = "rounded_test_total.txt"

    with open(output_file, "w") as file:
        file.write("Rounded Test Labels:\n")
        file.write(", ".join(map(str, rounded_test_labels)) + "\n\n")
        file.write("Rounded Test Predictions:\n")
        file.write(", ".join(map(str, rounded_test_preds)) + "\n")
    
    test_accuracy = accuracy_score(rounded_test_labels, rounded_test_preds)
    test_precision = precision_score(rounded_test_labels, rounded_test_preds, average="weighted", zero_division=1)
    test_recall = recall_score(rounded_test_labels, rounded_test_preds, average="weighted", zero_division=1)
    test_f1 = f1_score(rounded_test_labels, rounded_test_preds, average="weighted", zero_division=1)

    # Initialize arrays for each category (7 values per row)
    sentiment_test_labels = np.zeros(len(test_loader), dtype=float)
    happiness_test_labels  = np.zeros(len(test_loader), dtype=float)
    sadness_test_labels  = np.zeros(len(test_loader), dtype=float)
    anger_test_labels  = np.zeros(len(test_loader), dtype=float)
    disgust_test_labels  = np.zeros(len(test_loader), dtype=float)
    surprise_test_labels  = np.zeros(len(test_loader), dtype=float)
    fear_test_labels  = np.zeros(len(test_loader), dtype=float)

    for i in range(len(test_loader)):  # Loop through each "row" of values
        n = i * 7  # Start at the correct index for each row
        if(n+6)>rounded_test_labels.size:
            break
        
        sentiment_test_labels[i] = rounded_test_labels[n]
        happiness_test_labels[i] = rounded_test_labels[n + 1]
        sadness_test_labels[i] = rounded_test_labels[n + 2]
        anger_test_labels[i] = rounded_test_labels[n + 3]
        surprise_test_labels[i] = rounded_test_labels[n + 4]
        disgust_test_labels[i] = rounded_test_labels[n + 5]
        fear_test_labels[i] = rounded_test_labels[n + 6]

    sentiment_test_preds = np.zeros(len(test_loader), dtype=float)
    happiness_test_preds  = np.zeros(len(test_loader), dtype=float)
    sadness_test_preds  = np.zeros(len(test_loader), dtype=float)
    anger_test_preds  = np.zeros(len(test_loader), dtype=float)
    disgust_test_preds  = np.zeros(len(test_loader), dtype=float)
    surprise_test_preds  = np.zeros(len(test_loader), dtype=float)
    fear_test_preds  = np.zeros(len(test_loader), dtype=float)

    for i in range(len(test_loader)):  # Loop through each "row" of values
        n = i * 7  # Start at the correct index for each row
        if(n+6)>rounded_test_preds.size:
            break
        
        sentiment_test_preds[i] = rounded_test_preds[n]
        happiness_test_preds[i] = rounded_test_preds[n + 1]
        sadness_test_preds[i] = rounded_test_preds[n + 2]
        anger_test_preds[i] = rounded_test_preds[n + 3]
        surprise_test_preds[i] = rounded_test_preds[n + 4]
        disgust_test_preds[i] = rounded_test_preds[n + 5]
        fear_test_preds[i] = rounded_test_preds[n + 6]

    sentiment_test_accuracy = accuracy_score(sentiment_test_labels, sentiment_test_preds)
    happiness_test_accuracy = accuracy_score(happiness_test_labels, happiness_test_preds)
    sadness_test_accuracy = accuracy_score(sadness_test_labels, sadness_test_preds)
    anger_test_accuracy = accuracy_score(anger_test_labels, anger_test_preds)
    disgust_test_accuracy = accuracy_score(disgust_test_labels, disgust_test_preds)
    surprise_test_accuracy = accuracy_score(surprise_test_labels, surprise_test_preds)
    fear_test_accuracy = accuracy_score(fear_test_labels, fear_test_preds)

    sentiment_test_precision = precision_score(sentiment_test_labels, sentiment_test_preds, average="weighted", zero_division=1)
    sentiment_test_recall = recall_score(sentiment_test_labels, sentiment_test_preds, average="weighted", zero_division=1)
    sentiment_test_f1 = f1_score(sentiment_test_labels, sentiment_test_preds, average="weighted", zero_division=1)

    happiness_test_precision = precision_score(happiness_test_labels, happiness_test_preds, average="weighted", zero_division=1)
    happiness_test_recall = recall_score(happiness_test_labels, happiness_test_preds, average="weighted", zero_division=1)
    happiness_test_f1 = f1_score(happiness_test_labels, happiness_test_preds, average="weighted", zero_division=1)

    sadness_test_precision = precision_score(sadness_test_labels, sadness_test_preds, average="weighted", zero_division=1)
    sadness_test_recall = recall_score(sadness_test_labels, sadness_test_preds, average="weighted", zero_division=1)
    sadness_test_f1 = f1_score(sadness_test_labels, sadness_test_preds, average="weighted", zero_division=1)

    anger_test_precision = precision_score(anger_test_labels, anger_test_preds, average="weighted", zero_division=1)
    anger_test_recall = recall_score(anger_test_labels, anger_test_preds, average="weighted", zero_division=1)
    anger_test_f1 = f1_score(anger_test_labels, anger_test_preds, average="weighted", zero_division=1)

    disgust_test_precision = precision_score(disgust_test_labels, disgust_test_preds, average="weighted", zero_division=1)
    disgust_test_recall = recall_score(disgust_test_labels, disgust_test_preds, average="weighted", zero_division=1)
    disgust_test_f1 = f1_score(disgust_test_labels, disgust_test_preds, average="weighted", zero_division=1)

    surprise_test_precision = precision_score(surprise_test_labels, surprise_test_preds, average="weighted", zero_division=1)
    surprise_test_recall = recall_score(surprise_test_labels, surprise_test_preds, average="weighted", zero_division=1)
    surprise_test_f1 = f1_score(surprise_test_labels, surprise_test_preds, average="weighted", zero_division=1)

    fear_test_precision = precision_score(fear_test_labels, fear_test_preds, average="weighted", zero_division=1)
    fear_test_recall = recall_score(fear_test_labels, fear_test_preds, average="weighted", zero_division=1)
    fear_test_f1 = f1_score(fear_test_labels, fear_test_preds, average="weighted", zero_division=1)

    print("\n### Testing Metrics ###")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"\nSentiment Test Accuracy: {sentiment_test_accuracy:.4f}")
    print(f"Sentiment Test Precision: {sentiment_test_precision:.4f}")
    print(f"Sentiment Test Recall: {sentiment_test_recall:.4f}")
    print(f"Sentiment Test F1-Score: {sentiment_test_f1:.4f}")
    maek = mean_absolute_error(sentiment_test_labels, sentiment_test_preds)
    print(f"Sentiment Test MAE: {maek:.4f}")

    print(f"\nHappiness Test Accuracy: {happiness_test_accuracy:.4f}")
    print(f"Happiness Test Precision: {happiness_test_precision:.4f}")
    print(f"Happiness Test Recall: {happiness_test_recall:.4f}")
    print(f"Happiness Test F1-Score: {happiness_test_f1:.4f}")

    print(f"\nSadness Test Accuracy: {sadness_test_accuracy:.4f}")
    print(f"Sadness Test Precision: {sadness_test_precision:.4f}")
    print(f"Sadness Test Recall: {sadness_test_recall:.4f}")
    print(f"Sadness Test F1-Score: {sadness_test_f1:.4f}")

    print(f"\nAnger Test Accuracy: {anger_test_accuracy:.4f}")
    print(f"Anger Test Precision: {anger_test_precision:.4f}")
    print(f"Anger Test Recall: {anger_test_recall:.4f}")
    print(f"Anger Test F1-Score: {anger_test_f1:.4f}")

    print(f"\nDisgust Test Accuracy: {disgust_test_accuracy:.4f}")
    print(f"Disgust Test Precision: {disgust_test_precision:.4f}")
    print(f"Disgust Test Recall: {disgust_test_recall:.4f}")
    print(f"Disgust Test F1-Score: {disgust_test_f1:.4f}")

    print(f"\nSurprise Test Accuracy: {surprise_test_accuracy:.4f}")
    print(f"Surprise Test Precision: {surprise_test_precision:.4f}")
    print(f"Surprise Test Recall: {surprise_test_recall:.4f}")
    print(f"Surprise Test F1-Score: {surprise_test_f1:.4f}")

    print(f"\nFear Test Accuracy: {fear_test_accuracy:.4f}")
    print(f"Fear Test Precision: {fear_test_precision:.4f}")
    print(f"Fear Test Recall: {fear_test_recall:.4f}")
    print(f"Fear Test F1-Score: {fear_test_f1:.4f}")

#####################################################################################

# Load data
print("Loading datasets...")
language_train = np.load("processed_data/train_language.npy")  # (16327, 50, 300)
vision_train = np.load("processed_data/train_vision.npy")  # (16327, 50, 746)
acoustic_train = np.load("processed_data/train_acoustic.npy")  # (16327, 50, 74)
labels_train = np.load("processed_data/train_labels.npy")  # (16327, 1, 7)

language_test = np.load("processed_data/test_language.npy")  # (6533, 50, 300)
vision_test = np.load("processed_data/test_vision.npy")  # (6533, 50, 746)
acoustic_test = np.load("processed_data/test_acoustic.npy")  # (6533, 50, 74)
labels_test = np.load("processed_data/test_labels.npy")  # (6533, 1, 7)

# Prepare DataLoaders
train_loader = DataLoader(
    list(zip(language_train, vision_train, acoustic_train, labels_train)), batch_size=32, shuffle=True
)
test_loader = DataLoader(
    list(zip(language_test, vision_test, acoustic_test, labels_test)), batch_size=32, shuffle=False
)

# Train and evaluate the model
input_dimensions = [300, 746, 74]  # Language, vision, acoustic dimensions
train_and_evaluate_dfg_model(train_loader, test_loader, input_dimensions)
