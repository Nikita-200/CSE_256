#code to plot graphs for data
import numpy as np
import matplotlib.pyplot as plt

# Sample data (only Train Acc for all three models)
emotions = ['Sentiment', 'Happy', 'Sad', 'Anger', 'Surprise', 'Disgust', 'Fear']
metrics = ['Test F1']
model1 = np.array([
    [50.3],  # Test F1 for Sentiment
    [73.1],  # Test F1 for Happy
    [83.0],  # Test F1 for Sad
    [80.4],  # Test F1 for Anger
    [97.0],  # Test F1 for Surprise
    [85.9],  # Test F1 for Disgust
    [94.1]   # Test F1 for Fear
])
model2 = np.array([
    [43.6],  # Test F1 for Sentiment
    [63.3],  # Test F1 for Happy
    [79.3],  # Test F1 for Sad
    [79.4],  # Test F1 for Anger
    [97.0],  # Test F1 for Surprise
    [85.1],  # Test F1 for Disgust
    [94.1]   # Test F1 for Fear
])
model3 = np.array([
    [42.5],  # Test F1 for Sentiment
    [66.2],  # Test F1 for Happy
    [79.3],  # TTest F1 for Sad
    [78.6],  # Test F1 for Anger
    [97.0],  # Test F1 for Surprise
    [83.5],  # Test F1 for Disgust
    [94.1]   # TTest F1 for Fear
])

x = np.arange(len(emotions))  # Positions for emotions
bar_width = 0.25  # Width of bars (slightly smaller to fit all three models)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting Test F1 for model1
ax.bar(x - bar_width, model1[:, 0], bar_width, label='My DFG Test F1', alpha=0.7)

# Plotting Test F1 for model2
ax.bar(x, model2[:, 0], bar_width, label='Baseline 1 Test F1', alpha=0.7)

# Plotting Test F1 for model3
ax.bar(x + bar_width, model3[:, 0], bar_width, label='Baseline 2 Test F1', alpha=0.7)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(emotions)
ax.set_ylabel('Test Accuracy')
ax.set_title('Comparison of Test F1 Score for Sentiment & Emotion')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.tight_layout()

# Save the plot
plt.savefig("f1test_comparison_with_model3.png")

###############################
import numpy as np
import matplotlib.pyplot as plt

# Example data: Replace these with your actual counts
emotions = ['Sentiment','Happy', 'Sad', 'Anger', 'Surprise', 'Disgust', 'Fear']
matches = [3091, 4115, 5574, 5617, 6347, 5938, 6305]    # True positives
mismatches = [3442, 2418, 959, 916, 186, 595, 228]      # False positives/negatives

# X-axis positions
x = np.arange(len(emotions))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Stacked bars
bars1 = ax.bar(x, matches, label='Matches (True Positives)', color='skyblue')
bars2 = ax.bar(x, mismatches, bottom=matches, label='Mismatches (False Positives/Negatives)', color='salmon')

# Add annotations for true positives (matches)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 100,   # Position above the bar
            f'{int(height)}', ha='center', va='bottom', fontsize=9, color='black')

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(emotions)
ax.set_ylabel('Counts')
ax.set_title('Matches vs. Mismatches for Each Emotion')
ax.legend()

# Show plot
plt.tight_layout()
plt.savefig("Match_vs_Mismatch_on_DFG_annotated.png")
# plt.show()
