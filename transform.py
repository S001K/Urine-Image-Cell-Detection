import os
import pandas as pd
from PIL import Image

# Class mappings provided directly
class_mapping = {
    "pus": 0,
    "rbc": 1,
    "ep": 2,
}

def convert_csv_to_yolo(data_csv_path, images_dir, output_dir, class_mapping):
    """
    Convert dataset annotations from CSV to YOLO format.

    Args:
        data_csv_path (str): Path to train/test/validate CSV.
        images_dir (str): Path to image directory.
        output_dir (str): Path to save YOLO annotations.
        class_mapping (dict): Mapping of labels to class IDs.
    """
    # Create output subdirectories for images and labels
    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # Load dataset annotations
    df = pd.read_csv(data_csv_path)

    # Normalize labels (map point variants to main classes)
    df['label'] = df['label'].replace({
        "point-pus": "pus",
        "point-rbc": "rbc",
        "point-ep": "ep",
    })

    # Filter out labels not in class_mapping (e.g., missed class labels)
    df = df[df['label'].isin(class_mapping.keys())]

    for image_name in df['image'].unique():
        # Filter annotations for the image
        image_data = df[df['image'] == image_name]

        # Copy the image to the images output directory
        img_path = os.path.join(images_dir, image_name)
        output_img_path = os.path.join(images_output_dir, image_name)

        try:
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            with Image.open(img_path) as img:
                img.save(output_img_path)
            img_width, img_height = img.size
        except FileNotFoundError:
            print(f"Image {img_path} not found, skipping.")
            continue

        # Create YOLO annotation file in the labels folder
        yolo_file_path = os.path.join(labels_output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(yolo_file_path, "w") as yolo_file:
            for _, row in image_data.iterrows():
                xmin, ymin, xmax, ymax, label = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['label']
                class_id = class_mapping[label]

                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Images and annotations saved in {output_dir}")

# Main Function
def process_datasets(train_csv, test_csv, validate_csv, images_dir, output_dir):
    # Process each dataset
    for dataset_csv, dataset_name in zip(
        [train_csv, test_csv, validate_csv], ['train', 'test', 'validate']
    ):
        print(f"Processing {dataset_name} dataset...")
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        convert_csv_to_yolo(dataset_csv, images_dir, dataset_output_dir, class_mapping)

# Usage
train_csv = "UMID-Urine-Microscopic-Image-Dataset-main/train.csv"
test_csv = "UMID-Urine-Microscopic-Image-Dataset-main/test.csv"
validate_csv = "UMID-Urine-Microscopic-Image-Dataset-main/val.csv"
images_dir = "UMID-Urine-Microscopic-Image-Dataset-main/images"  # Path to image directory
output_dir = "output2/"  # Base directory for datasets

process_datasets(train_csv, test_csv, validate_csv, images_dir, output_dir)
