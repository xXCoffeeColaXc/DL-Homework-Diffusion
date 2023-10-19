import config
import os
import json

def create_bbox_dict():
    with open(os.path.join(config.ROOT_DIR, "bounding_boxes.txt"), "r") as f:
        bounding_boxes_content = f.readlines()

    bounding_boxes_dict = {}
    for line in bounding_boxes_content:
        values = line.strip().split()
        img_id = int(values[0])
        bbox = tuple(map(float, values[1:]))
        bounding_boxes_dict[img_id] = bbox

    #return bounding_boxes_dict
    with open(os.path.join("data", "metadata", "bounding_boxes.json"), 'w') as json_file:
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
    
    #return classes_dict
    with open(os.path.join("data", "metadata", "classes.json"), 'w') as json_file:
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

    #return image_class_labels_dict
    with open(os.path.join("data", "metadata", "image_class_labels.json"), 'w') as json_file:
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
    
    #return images_dict
    with open(os.path.join("data", "metadata", "images.json"), 'w') as json_file:
        json.dump(images_dict, json_file, indent=4)

def create_split_dict():
    with open(os.path.join(config.ROOT_DIR, "train_test_split.txt"), "r") as f:
        train_test_split_content = f.readlines()

    train_test_split_dict = {}
    for line in train_test_split_content:
        values = line.strip().split()
        img_id = int(values[0])
        split = "train" if values[1] == "1" else "test"
        train_test_split_dict[img_id] = split

    #return train_test_split_dict
    with open(os.path.join("data", "metadata", "train_test_split.json"), 'w') as json_file:
        json.dump(train_test_split_dict, json_file, indent=4)


if __name__ == "__main__":
    create_bbox_dict()
    create_classes_dict()
    create_image_dict()
    create_labels_dict()
    create_split_dict()




