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
    def __init__(self, root_dir, metadata_dir, image_size, transform=None, mode="train", logging=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string): "train" or "test" to specify the dataset split.
        """
        self.root_dir = root_dir
        self.metadata_dir = metadata_dir
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.logging = logging
        
        # Load the dictionaries
        self.images_dict = json.load(open(os.path.join(metadata_dir, "images.json"), "r"))
        self.bounding_boxes_dict = json.load(open(os.path.join(metadata_dir, "bounding_boxes.json"), "r"))
        self.image_class_labels_dict = json.load(open(os.path.join(metadata_dir, "image_class_labels.json"), "r"))
        self.train_test_split_dict = json.load(open(os.path.join(metadata_dir, "train_test_split.json"), "r"))
        self.classes_dict = json.load(open(os.path.join(metadata_dir, "classes.json"), "r"))
        
        # Filter the image IDs based on the split
        self.image_ids = [img_id for img_id, s in self.train_test_split_dict.items() if s == self.mode]

    def __len__(self):
        #return len(self.image_ids) * 2 #increase the dataset size by 2x:
        return len(self.image_ids)

    def __getitem__(self, idx):
        #img_id = self.image_ids[idx]

        '''
        Each image in the dataset will be augmented and fetched twice in one epoch, 
        effectively doubling the dataset size.
        '''

        #img_id = self.image_ids[idx % len(self.image_ids)]  # wrap around to the original dataset
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
            target_width = self.image_size
            target_height = self.image_size
            
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
    
    def denormalize():
        pass

def get_data(image_dir, metadata_dir, image_size=64, batch_size=8, mode="train", num_workers=1):
    # Common transformations
    mean = torch.tensor([0.4865, 0.4998, 0.4323])
    std = torch.tensor([0.2326, 0.2276, 0.2659])
    base_transforms = [
        transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if mode == "train":
        # Some other transformations:
        #transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.2),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

        base_transforms.insert(2, transforms.RandomHorizontalFlip()) # Insert RandomHorizontalFlip after RandomCrop

    # TODO rethink base_transforms
    if mode == "test":
        base_transforms = [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    

    transform = transforms.Compose(base_transforms)

    dataset = CUB200Dataset(image_dir, metadata_dir, image_size, transform, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader
