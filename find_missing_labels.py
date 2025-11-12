import os

dataset_root = r"C:/Users/JuniorCHIEMMANUELNGU/Desktop/test/dataset"

def check_labels(set_name):
    images_dir = os.path.join(dataset_root, "images", set_name)
    labels_dir = os.path.join(dataset_root, "labels", set_name)
    image_files = set(f[:-4] for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png")))
    label_files = set(f[:-4] for f in os.listdir(labels_dir) if f.endswith(".txt"))
    
    missing_labels = image_files - label_files
    missing_images = label_files - image_files
    
    if missing_labels:
        print(f"Images in {set_name} without labels: {missing_labels}")
    else:
        print(f"All images in {set_name} have matching label files.")
    
    if missing_images:
        print(f"Label files in {set_name} without images: {missing_images}")
    else:
        print(f"All label files in {set_name} have matching images.")

check_labels("train")
check_labels("val")
