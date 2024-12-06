#code to read and display the content of csd files, npy files and to plot some graphs
import h5py
import numpy as np
import itertools

# Load CSD file and inspect structure
def inspect_csd(file_path):
    with h5py.File(file_path, 'r') as f:
        print("File structure:")
        f.visititems(lambda name, obj: print(name, "->", obj))
        return f  # Keep the file open to explore interactively
    
def load_csd(file_path):
    with h5py.File(file_path, 'r') as f:
        data = {key: np.array(f[key]) for key in f.keys()}
    return data

def dataset_features_and_intervals(file_path, key_name):
    with h5py.File(file_path, "r") as f:
        # Navigate to the 'All Labels' computational sequence
        all_labels_group = f[key_name]["data"]

        # Iterate through each entry in the group
        #for key in itertools.islice(all_labels_group.keys(), 1):
        for key in all_labels_group.keys():
            print(f"Key: {key}")

            # Access the datasets inside each key
            group = all_labels_group[key]
            for dataset_name in group.keys():
                dataset = group[dataset_name]
                
                # Check if the dataset contains data
                if isinstance(dataset, h5py.Dataset):
                    print(f"Dataset Name: {dataset_name}")
                    print(f"Values: {dataset[()]}")  # Access the actual data
                    
            print("-" * 50)

        # Optionally explore metadata for additional information
        metadata = f[key_name]["metadata"]
        for key in metadata.keys():
            print(f"Metadata - {key}: {metadata[key][()]}")

# ################################# Open and inspect the GloVe vectors CSD file
# file_path = "CMU_MOSEI_TimestampedWordVectors.csd"
# print("CMU_MOSEI_TimestampedWordVectors.csd")
# dataset_features_and_intervals(file_path, key_name='glove_vectors')

# ########################################COVAREP
# file_path = "CMU_MOSEI_COVAREP.csd"
# print("CMU_MOSEI_COVAREP.csd")
# dataset_features_and_intervals(file_path, key_name='COVAREP')

# ##########################VISUAL FACET 42
# file_path = "CMU_MOSEI_VisualFacet42.csd"
# print("CMU_MOSEI_VisualFacet42.csd")
# dataset_features_and_intervals(file_path, key_name='FACET 4.2')

# ################################Open Face2
# file_path = "CMU_MOSEI_VisualOpenFace2.csd"
# print("CMU_MOSEI_VisualOpenFace2.csd")
# dataset_features_and_intervals(file_path, key_name='OpenFace_2')

# #############################LABELS
# file_path = "CMU_MOSEI_Labels.csd"
# print("CMU_MOSEI_Labels.csd")
# dataset_features_and_intervals(file_path, key_name='All Labels')

# file_path = "final_aligned\\All Labels.csd"
# dataset_features_and_intervals(file_path, key_name='All Labels')

# To display the npy file descriptors
# import numpy as np

# file_path = "processed_data/train_acoustic.npy"
# data = np.load(file_path)
# #Print out the details
# print("Data Type:", type(data))
# print("Shape:", data.shape)
# print("First 5 Entries:\n", data[:5])  # Display the first 10 entries


# Read the file and extract relevant parts
with open("rounded_test_total.txt", "r") as file:
    lines = file.readlines()

# Extract the data after the respective labels
rounded_labels = []
rounded_predictions = []

# Check where the lines for Rounded Train Labels and Rounded Train Predictions start
for idx, line in enumerate(lines):
    if line.strip() == "Rounded Test Labels:":
        # Extract the next line after the label header
        rounded_labels = lines[idx + 1].strip()
        integer_list = [int(float(value)) for value in rounded_labels.split(', ')]
        rounded_labels = integer_list

    elif line.strip() == "Rounded Test Predictions:":
        # Extract the next line after the predictions header
        rounded_predictions = lines[idx + 1].strip()
        integer_list = [int(float(value)) for value in rounded_predictions.split(', ')]
        rounded_predictions = integer_list

# Task 1: Calculate the number of matched and unmatched values for sentiment
sentiment_labels = rounded_labels[0::7]
sentiment_preds = rounded_predictions[0::7]
matched_sentiment = sum([1 for i in range(len(sentiment_labels)) if sentiment_labels[i] == sentiment_preds[i]])
unmatched_sentiment = len(sentiment_labels) - matched_sentiment

# Task 2: Calculate the number of matched and unmatched values for happiness
happiness_labels = rounded_labels[1::7]
happiness_preds = rounded_predictions[1::7]
matched_happiness = sum([1 for i in range(len(happiness_labels)) if happiness_labels[i] == happiness_preds[i]])
unmatched_happiness = len(happiness_labels) - matched_happiness

# Task 3: Calculate the number of matched and unmatched values for sadness
sadness_labels = rounded_labels[2::7]
sadness_preds = rounded_predictions[2::7]
matched_sadness = sum([1 for i in range(len(sadness_labels)) if sadness_labels[i] == sadness_preds[i]])
unmatched_sadness = len(sadness_labels) - matched_sadness

# Task 4: Calculate the number of matched and unmatched values for anger
anger_labels = rounded_labels[3::7]
anger_preds = rounded_predictions[3::7]
matched_anger = sum([1 for i in range(len(anger_labels)) if anger_labels[i] == anger_preds[i]])
unmatched_anger = len(anger_labels) - matched_anger

# Task 5: Calculate the number of matched and unmatched values for surprise
surprise_labels = rounded_labels[4::7]
surprise_preds = rounded_predictions[4::7]
matched_surprise = sum([1 for i in range(len(surprise_labels)) if surprise_labels[i] == surprise_preds[i]])
unmatched_surprise = len(surprise_labels) - matched_surprise

# Task 6: Calculate the number of matched and unmatched values for disgust
disgust_labels = rounded_labels[5::7]
disgust_preds = rounded_predictions[5::7]
matched_disgust = sum([1 for i in range(len(disgust_labels)) if disgust_labels[i] == disgust_preds[i]])
unmatched_disgust = len(disgust_labels) - matched_disgust

# Task 7: Calculate the number of matched and unmatched values for fear
fear_labels = rounded_labels[6::7]
fear_preds = rounded_predictions[6::7]
matched_fear = sum([1 for i in range(len(fear_labels)) if fear_labels[i] == fear_preds[i]])
unmatched_fear = len(fear_labels) - matched_fear
print("For Test Data:")
# Print the results
print(f"Sentiment - matched: {matched_sentiment}, unmatched: {unmatched_sentiment}")
print(f"Happiness - matched: {matched_happiness}, unmatched: {unmatched_happiness}")
print(f"Sadness - matched: {matched_sadness}, unmatched: {unmatched_sadness}")
print(f"Anger - matched: {matched_anger}, unmatched: {unmatched_anger}")
print(f"Surprise - matched: {matched_surprise}, unmatched: {unmatched_surprise}")
print(f"Disgust - matched: {matched_disgust}, unmatched: {unmatched_disgust}")
print(f"Fear - matched: {matched_fear}, unmatched: {unmatched_fear}")


##### Graph Plots
import matplotlib.pyplot as plt
# emotions = ["Happiness","Sadness","Anger","Surprise","Disgust","Fear"]
emotions = ["Sentiment"]
counts = {emotion: {} for emotion in emotions}
# Correcting the range extraction logic
ranges = ["-3 to -2.5","-2.5 to -1.5", "-1.5 to -0.5", "-0.5 to 0.5", "0.5 to 1.5", "1.5 to 2.5", "2.5 to 3"]
counts["Sentiment"]["-3 to -2.5"] = 66
counts["Sentiment"]["-2.5 to -1.5"] = 509
counts["Sentiment"]["-1.5 to -0.5"] = 800
counts["Sentiment"]["-0.5 to 0.5"] = 2760
counts["Sentiment"]["0.5 to 1.5"] = 1703
counts["Sentiment"]["1.5 to 2.5"] = 607
counts["Sentiment"]["2.5 to 3"] = 60

# counts["Happiness"]["0 to 0.5"] = 1302
# counts["Happiness"]["0.5 to 1.5"] = 1663
# counts["Happiness"]["1.5 to 2.5"] = 520
# counts["Happiness"]["2.5 to 3"] = 25

# counts["Sadness"]["0 to 0.5"] = 797
# counts["Sadness"]["0.5 to 1.5"] = 819
# counts["Sadness"]["1.5 to 2.5"] = 32
# counts["Sadness"]["2.5 to 3"] = 1

# counts["Anger"]["0 to 0.5"] = 545
# counts["Anger"]["0.5 to 1.5"] = 802
# counts["Anger"]["1.5 to 2.5"] = 57
# counts["Anger"]["2.5 to 3"] = 5

# counts["Surprise"]["0 to 0.5"] = 458
# counts["Surprise"]["0.5 to 1.5"] = 183
# counts["Surprise"]["1.5 to 2.5"] = 3
# counts["Surprise"]["2.5 to 3"] = 0

# counts["Disgust"]["0 to 0.5"] = 534
# counts["Disgust"]["0.5 to 1.5"] = 493
# counts["Disgust"]["1.5 to 2.5"] = 52
# counts["Disgust"]["2.5 to 3"] = 7

# counts["Fear"]["0 to 0.5"] = 333
# counts["Fear"]["0.5 to 1.5"] = 228
# counts["Fear"]["1.5 to 2.5"] = 0
# counts["Fear"]["2.5 to 3"] = 0

plot_data = {rng: [counts[emotion][rng] for emotion in emotions] for rng in ranges}

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2  # Width for bar plots
x = np.arange(len(emotions))  # X positions for the groups

legend_labels = ['-3','-2','-1','0', '1', '2', '3']
# Create a bar for each range
for i, rng in enumerate(ranges):
    ax.bar(x + i * width, plot_data[rng], width, label=legend_labels[i])

# Setting labels, ticks, and legend
ax.set_ylabel("Counts")
ax.set_xlabel("Sentiment")
ax.set_title("Sentiment Counts by Value Ranges")
ax.set_xticks(x + width * (len(ranges) - 1) / 2)
ax.set_xticklabels(emotions)
ax.legend(title="Value Ranges")

# # Display the plot
plt.tight_layout()
plt.savefig("Sentiment Count for Test Set.png")