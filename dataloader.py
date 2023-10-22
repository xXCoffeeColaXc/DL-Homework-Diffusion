from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json 
import config
from torchvision.utils import save_image
import numpy as np  
import torch

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train", logging=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string): "train" or "test" to specify the dataset split.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.logging = logging
        
        # Load the dictionaries
        self.images_dict = json.load(open(os.path.join(config.METADATA_DIR, "images.json"), "r"))
        self.bounding_boxes_dict = json.load(open(os.path.join(config.METADATA_DIR, "bounding_boxes.json"), "r"))
        self.image_class_labels_dict = json.load(open(os.path.join(config.METADATA_DIR, "image_class_labels.json"), "r"))
        self.train_test_split_dict = json.load(open(os.path.join(config.METADATA_DIR, "train_test_split.json"), "r"))
        self.classes_dict = json.load(open(os.path.join(config.METADATA_DIR, "classes.json"), "r"))
        
        # Filter the image IDs based on the split
        self.image_ids = [img_id for img_id, s in self.train_test_split_dict.items() if s == self.split]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, "images", self.images_dict[img_id])
        if self.logging:
            print(img_path)
        image = Image.open(img_path).convert("RGB")
        # Get the original image dimensions
        orig_width, orig_height = image.size
        # Get the corresponding label
        label = self.image_class_labels_dict[img_id]

        str_label = self.classes_dict[str(label)]
        if self.logging:
            print(str_label)
        
        # Get the bounding box (if needed)
        bbox = self.bounding_boxes_dict[img_id].copy()
        
        # Assuming you're resizing the image to (config.IMAGE_SIZE, config.IMAGE_SIZE)
        if self.transform:
            target_width = config.IMAGE_SIZE
            target_height = config.IMAGE_SIZE
            
            # Calculate scaling factors
            x_scale = target_width / orig_width
            y_scale = target_height / orig_height
            
            # Adjust the bounding box
            bbox[0] = bbox[0] * x_scale  # x
            bbox[1] = bbox[1] * y_scale  # y
            bbox[2] = bbox[2] * x_scale  # width
            bbox[3] = bbox[3] * y_scale  # height
        bbox = torch.tensor(bbox)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label, bbox
    

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # idk 
        
        transforms.Normalize(0.5, 0.5)
    ])


    dataset = CUB200Dataset(config.ROOT_DIR, transform)
    loader = DataLoader(dataset, batch_size=5)
    for idx, (x,y) in enumerate(loader):
        print(x.shape)
        save_image(x * 0.5 + 0.5, f"x{idx}.png")

        
        if idx == 3:
            import sys

            sys.exit()