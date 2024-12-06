import os
import numpy as np
from mmsdk import mmdatasdk
from torch.utils.data import Dataset
import torch
from standard_split import standard_train_fold, standard_test_fold

highlevel = {
    "glove_vectors": "final_aligned\glove_vectors.csd",
    "COAVAREP": "final_aligned\COAVAREP.csd",
    "OpenFace_2": "final_aligned\OpenFace_2.0.csd",
    "FACET 4.2": "final_aligned\FACET 4.2.csd",
}

labels = {"All Labels": "final_aligned\All Labels.csd"}

cmu_mosei_modalities = {
    "highlevel": highlevel,
    "labels": labels
}


def process_sequence(seq, seq_len):
    """Pad or trim sequence to fixed length."""
    if len(seq) > seq_len:
        return seq[:seq_len]
    else:
        return np.pad(seq, ((0, seq_len - len(seq)), (0, 0)), mode="constant")

def deploy(in_dataset,destination):
	deploy_files={x:x for x in in_dataset.keys()}
	in_dataset.deploy(destination,deploy_files)

def preprocess_mosei(data_path, save_dir, seq_len=50):
    os.makedirs(save_dir, exist_ok=True)

    cmumosei_dataset={}
    # print("Loading CMU-MOSEI dataset...")
    cmumosei_dataset["highlevel"] = mmdatasdk.mmdataset(cmu_mosei_modalities["highlevel"], data_path)
    cmumosei_dataset["labels"] = mmdatasdk.mmdataset(cmu_mosei_modalities["labels"], data_path)
    #################################################################################################################
    ''' Commented out code for building final_align directory since it takes about 2 to 3 days '''
    # # Align modalities to "glove_vectors" and handle missing data
    # cmumosei_dataset["highlevel"].align("glove_vectors") #aligns all sequences to a reference computational sequence, with optional collapsing of features.
    # cmumosei_dataset["highlevel"].impute("glove_vectors")  # Impute missing data in "glove_vectors" #fills in missing data using a specified imputation function.

    # # Save word-aligned computational sequences
    # deploy(cmumosei_dataset["highlevel"], "word_aligned_highlevel")

    # # Add labels to the dataset
    # cmumosei_dataset["highlevel"].computational_sequences["All Labels"] = cmumosei_dataset["labels"]["All Labels"]

    # # Align all data to labels
    # cmumosei_dataset["highlevel"].align("All Labels")

    # # Remove sequences with missing modality information # performs a stricter version of unification, removing entries that don’t match in all sequences.
    # cmumosei_dataset["highlevel"].hard_unify()

    # # Save the final aligned dataset
    # deploy(cmumosei_dataset["highlevel"], "final_aligned") #aves the computational sequences to a specified destination folder, associating each sequence with a specific filename.
    #######################################################################################################################
    test_keys = [key for key in standard_test_fold]
    train_keys = [key for key in standard_train_fold]
    folds = [train_keys, test_keys]

    # Create tensors for train and test splits
    highlevel_tensors = cmumosei_dataset["highlevel"].get_tensors( #converts the dataset into tensors, with optional padding and fold splitting
        seq_len=seq_len,
        non_sequences=[],
        direction=False,
        folds=folds
    )
    labels_tensors = cmumosei_dataset["labels"].get_tensors(
        seq_len=seq_len,
        non_sequences=["All Labels"],  # Labels are non-sequential
        direction=False,
        folds=folds
    )

    fold_names = ["train", "test"]

    print("HIGH LEVEL KEYS=",cmumosei_dataset["highlevel"].keys())
    print("LABEL KEYS=",cmumosei_dataset["labels"].keys())

    for i, fold_name in enumerate(fold_names):
        print(f"Keys in highlevel_tensors[{fold_name}]:", highlevel_tensors[i].keys())
        print(f"Keys in labels_tensors[{fold_name}]:", labels_tensors[i].keys())

        # Extract modality tensors
        openface = highlevel_tensors[i]["OpenFace_2"]
        facet = highlevel_tensors[i]["FACET 4.2"]
        vision = np.concatenate([openface, facet], axis=-1)

        language = highlevel_tensors[i]["glove_vectors"]
        acoustic = highlevel_tensors[i]["COAVAREP"]
        labels = labels_tensors[i]["All Labels"]

        # Save modality tensors and labels
        np.save(os.path.join(save_dir, f"{fold_name}_language.npy"), language)
        np.save(os.path.join(save_dir, f"{fold_name}_vision.npy"), vision)
        np.save(os.path.join(save_dir, f"{fold_name}_acoustic.npy"), acoustic)
        np.save(os.path.join(save_dir, f"{fold_name}_labels.npy"), labels)

        # Output tensor shapes
        print(f"Shape of language tensor for {fold_name} fold: {language.shape}")
        print(f"Shape of vision tensor for {fold_name} fold: {vision.shape}")
        print(f"Shape of acoustic tensor for {fold_name} fold: {acoustic.shape}")
        print(f"Shape of labels tensor for {fold_name} fold: {labels.shape}")

    print("Data preprocessing complete!")


# PyTorch Dataset Class
class MOSEIDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.language = np.load(os.path.join(data_dir, f"{split}_language.npy"))
        self.vision = np.load(os.path.join(data_dir, f"{split}_vision.npy"))
        self.acoustic = np.load(os.path.join(data_dir, f"{split}_acoustic.npy"))
        self.labels = np.load(os.path.join(data_dir, f"{split}_labels.npy"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.language[idx], dtype=torch.float32),
            torch.tensor(self.vision[idx], dtype=torch.float32),
            torch.tensor(self.acoustic[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# Example Usage
if __name__ == "__main__":
    # Paths
    data_path = "CMU_MOSEI"
    save_dir = "processed_data"
    seq_len = 50

    # Preprocess and save data
    preprocess_mosei(data_path, save_dir, seq_len=seq_len)
