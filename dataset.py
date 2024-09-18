import os
import re
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# カスタムデータセットの定義

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, mode, class_idx, transform=None, load_data=None):
        self.transform = transform
        self.data = []

        if mode == "full":
            pattern = re.compile(r'n[0-9]{8}')
            for folder_name in sorted(filter(pattern.fullmatch, os.listdir(root_dir))):
                folder_path = os.path.join(root_dir, folder_name, 'images')
                image_files = sorted(os.listdir(folder_path))

                for file in image_files:
                    self.data.append((os.path.join(folder_path, file), class_idx[folder_name]))

        elif mode in ["train", "val"]:
            if load_data is not None:
                self.data = load_data
            else:
                raise ValueError(f"Data must be provided for 'train' or 'val' mode, but 'load_data' was not supplied.")


        elif mode == "test":
            df = pd.read_csv(os.path.join(root_dir, "val_annotations.txt"), sep=r'\s+', header=None)
            for image_file in sorted(os.listdir(os.path.join(root_dir, 'images'))):
                image_label = df[df[0] == image_file][1].iloc[0]
                self.data.append((os.path.join(root_dir, "images", image_file), class_idx[image_label]))


        elif mode == "submission":
            image_files = sorted(os.listdir(os.path.join(root_dir, 'images')))
            for image_file in image_files:
                self.data.append((os.path.join(root_dir, "images", image_file), None))


        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of:  'full', 'train', 'val', 'test', 'submission.")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
