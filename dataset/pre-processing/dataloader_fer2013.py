# IMPORT LIBABRIES
""" Prerequisites: python envirnoment -- mindspore installed".
    This datasetloader can be directly plugged into the model (CNN training loop or whichever model is trained ).
"""
import mindspore.dataset as ds
import mindspore.dataset.vision as vision # For image augmentations and transformations
from mindspore.dataset import GeneratorDataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Custom FER2013 Dataset
# -----------------------------
class FERDataset:
    """
    Loader for the FER2013_split dataset.
    - Images: 48x48 grayscale
    - Labels: Official FER2013 mapping (0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral)
    - Supports train, validation, and test splits
    """

    def __init__(self, data_dir, split='train', img_size=48):
        # Set dataset path for the chosen split (train/val/test)
        self.data_dir = os.path.join(data_dir, split)
        self.img_size = img_size

        # Official FER2013 label mapping
        self.emotion_map = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
        }

        # Only keep directories that match official emotion labels -- this ensures system files like .DS_Store are ignored
        self.classes = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d in self.emotion_map
        )

        # Map class names to numeric labels
        self.class_to_idx = {cls: self.emotion_map[cls] for cls in self.classes}

        # Collect all image file paths with corresponding labels
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
         # Total number of samples in this dataset split
        return len(self.samples)

    # Redundant since the dataset is already pre-processed: grayscale
    def __getitem__(self, idx):
        # Get image and label for a given index
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')          # grayscale
        image = np.array(image, dtype=np.float32) / 255.0 # normalize
        # Expand dimensions to match MindSpore CNN input 
        image = np.expand_dims(image, axis=0)  # (C,H,W) = (1,48,48)
        return image, label

# -----------------------------
# Data Augmentation
# -----------------------------
def create_transforms(split='train'):
    """
        Returns a list of augmentation operations for MindSpore.

        - Training set: random horizontal flips and small affine transforms
        (helps the model generalize and reduces overfitting)
        - Validation/Test set: no augmentation, only raw images
    """
    if split == 'train':
        return [
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
    else:
        return []  # no augmentation for val/test

# -----------------------------
# DataLoader
# -----------------------------
def create_dataloader(data_dir, split='train', batch_size=32, shuffle=True):
    """
        Create a MindSpore GeneratorDataset with batching and optional augmentation.

        - data_dir: root folder of FER2013_split
        - split: 'train', 'val', or 'test'
        - batch_size: number of samples per batch
        - shuffle: shuffle data for training
    """

    # Load the custom dataset
    dataset = FERDataset(data_dir, split=split)

    # Create MindSpore GeneratorDataset
    ms_dataset = GeneratorDataset(
        source=dataset,
        column_names=['image', 'label'],
        shuffle=shuffle,
        num_parallel_workers=2  # Reduce parallel workers to avoid memory issues
    )

    # Apply transformations (augmentations)
    transforms = create_transforms(split)
    if transforms:
        ms_dataset = ms_dataset.map(
            operations=transforms,
            input_columns=['image']
        )

    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True)
    return ms_dataset


# Usage
dataset_root = 'FER2013_split'

# Create dataloaders for training and validation
train_loader = create_dataloader(dataset_root, 'train', batch_size=32, shuffle=True)
val_loader   = create_dataloader(dataset_root, 'val', batch_size=32, shuffle=False)

# verify batch shapes
if __name__ == "__main__":
    for batch in train_loader.create_dict_iterator():
        # MindSpore returns batches as dictionaries
        images = batch['image'] # shape: (batch_size, 1, 48, 48)
        labels = batch['label'] # shape: (batch_size,)
        print("Images batch shape:", images.shape)  
        print("Labels batch shape:", labels.shape)  
        print(labels[:5])
        break  # Only check the first batch

"""
DISPLAY A IMAGE 
        # Pick the first image in the batch
        first_image = images[0].asnumpy()   # convert MindSpore Tensor to NumPy
        first_image = first_image.squeeze() # remove channel dimension -> (48,48)

        # Show the image
        plt.imshow(first_image, cmap='gray')
        plt.title(f"Label: {labels[0].asnumpy()}")  # show emotion label
        plt.axis('off')
        plt.show()
"""



