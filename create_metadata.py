import config
import os
import json
import random

'''
bounding_boxes_dict: Key is the image ID and value is a tuple of four numbers representing the bounding box.
    Image 1: Bounding box (60.0, 27.0, 325.0, 304.0)
    Image 2: Bounding box (139.0, 30.0, 153.0, 264.0)

classes_dict: Key is the class ID and value is the class name.
    Class 1: 001.Black_footed_Albatross
    Class 2: 002.Laysan_Albatross

image_class_labels_dict: Key is the image ID and value is the class ID.
    Image 1: Class 1
    Image 2: Class 1

images_dict: Key is the image ID and value is the image path.
    Image 1: 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
    Image 2: 001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg

train_test_split_dict: Key is the image ID and value is 'train' or 'test' based on the binary value.
    Image 1: test
    Image 2: train
'''

def create_bbox_dict():
    with open(os.path.join(config.ROOT_DIR, "bounding_boxes.txt"), "r") as f:
        bounding_boxes_content = f.readlines()

    bounding_boxes_dict = {}
    for line in bounding_boxes_content:
        values = line.strip().split()
        img_id = int(values[0])
        bbox = tuple(map(float, values[1:]))
        bounding_boxes_dict[img_id] = bbox

    with open(os.path.join(config.METADATA_DIR, "bounding_boxes.json"), 'w') as json_file:
        json.dump(bounding_boxes_dict, json_file, indent=4)

def create_classes_dict():
    with open(os.path.join(config.ROOT_DIR, "classes.txt"), "r") as f:
        classes_content = f.readlines()

    classes_dict = {}
    for line in classes_content:
        values = line.strip().split()
        class_id = int(values[0])
        class_name = " ".join(values[1:])
        classes_dict[class_id] = class_name
    
    with open(os.path.join(config.METADATA_DIR, "classes.json"), 'w') as json_file:
        json.dump(classes_dict, json_file, indent=4)

def create_labels_dict():
    with open(os.path.join(config.ROOT_DIR, "image_class_labels.txt"), "r") as f:
        image_class_labels_content = f.readlines()

    image_class_labels_dict = {}
    for line in image_class_labels_content:
        values = line.strip().split()
        img_id = int(values[0])
        class_id = int(values[1])
        image_class_labels_dict[img_id] = class_id

    with open(os.path.join(config.METADATA_DIR, "image_class_labels.json"), 'w') as json_file:
        json.dump(image_class_labels_dict, json_file, indent=4)

def create_image_dict():
    with open(os.path.join(config.ROOT_DIR, "images.txt"), "r") as f:
        images_content = f.readlines()

    images_dict = {}
    for line in images_content:
        values = line.strip().split()
        img_id = int(values[0])
        img_path = values[1]
        images_dict[img_id] = img_path
    
    with open(os.path.join("data", "metadata", "images.json"), 'w') as json_file:
        json.dump(images_dict, json_file, indent=4)

def create_split_dict(split_ratio=0.8):
    total_images = len(open(os.path.join(config.ROOT_DIR, "train_test_split.txt")).readlines())
    
    num_train_samples = int(total_images * split_ratio)

    # Create a list with labels based on the split
    labels = ["train"] * num_train_samples + ["test"] * (total_images - num_train_samples)
    
    random.shuffle(labels)

    # Create a dictionary with IDs in order and the shuffled labels
    train_test_split_dict = {}
    for img_id, label in zip(range(1, total_images + 1), labels):
        train_test_split_dict[img_id] = label

    with open(os.path.join(config.METADATA_DIR, "train_test_split.json"), 'w') as json_file:
        json.dump(train_test_split_dict, json_file, indent=4)


if __name__ == "__main__":

    #create_bbox_dict()
    #create_classes_dict()
    #create_image_dict()
    #create_labels_dict()
    create_split_dict(split_ratio=0.8)




