#baseline model
'''A simple feed forward model that takes as input 300-dimensional Glove features (textual modality) and 
 predicts sentiment and emotion'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GloveDataset(Dataset):
    #the embeddings are the 300-d GloVe embeddings of a datapoint
    #labels are the 7-valued sentiment and emotion labels
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.mean(x, dim=1)  # Mean pooling across time steps (sequence length of 50)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load data
train_embeddings = np.load("processed_data/train_language.npy")
train_labels = np.load("processed_data/train_labels.npy")
test_embeddings = np.load("processed_data/test_language.npy")
test_labels = np.load("processed_data/test_labels.npy")

# Create datasets and dataloaders
batch_size = 32
train_dataset = GloveDataset(train_embeddings, train_labels)
test_dataset = GloveDataset(test_embeddings, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
input_dim = train_embeddings.shape[2]  # 300 for GloVe
hidden_dim = 128
output_dim = 7  # Number of classes
model = FeedforwardNN(input_dim, hidden_dim, output_dim)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 30
all_train_preds_total, all_train_labels_total = [], []
all_test_labels_total, all_test_preds_total = [], []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    train_preds, train_labels_epoch = [], []
    for embeddings, labels in train_loader:
        embeddings = embeddings.float() #embeddings [32,50,300]
        labels = labels.float().squeeze(1) #labels [32,7]
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        train_preds.extend((outputs).detach().numpy())
        train_labels_epoch.extend(labels.cpu().numpy())

    all_training_labels = np.array(train_labels_epoch).flatten() #flatten into a single layer
    all_training_preds = np.array(train_preds).flatten()
    rounded_labels = np.round(all_training_labels).astype(int)
    rounded_preds = np.round(all_training_preds).astype(int)

    train_acc = accuracy_score(rounded_labels, rounded_preds)
    print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")

# Evaluation
model.eval()
test_preds, test_labels_epoch = [], []
with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings)
        test_preds.extend((outputs).detach().numpy())
        all_test_preds_total.extend((outputs).detach().numpy())
        test_labels_epoch.extend(labels.cpu().numpy())
        all_test_labels_total.extend(labels.cpu().numpy())

    all_testing_labels = np.array(test_labels_epoch).flatten()
    all_testing_preds = np.array(test_preds).flatten()
    rounded_labels = np.round(all_testing_labels).astype(int)
    rounded_preds = np.round(all_testing_preds).astype(int)

test_acc = accuracy_score(rounded_labels, rounded_preds)
print(f"Test Accuracy: {test_acc:.4f}")


print("\nFinal Training and Testing Metrics:")

# Training Metrics
rounded_train_labels = np.round(all_training_labels).astype(int)
rounded_train_preds = np.round(all_training_preds).astype(int)
output_file = "rounded_train_total_baseline1.txt"

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
output_file = "rounded_test_total_baseline1.txt"

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

# Process the values in A in chunks of 7 values
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

# Process the values in A in chunks of 7 values
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
