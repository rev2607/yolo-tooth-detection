#!/usr/bin/env python3
"""
YOLO Tooth Detection Training Pipeline
Converted from Jupyter notebook
"""

import os
import random
import shutil
import subprocess
import sys

def run_command(command):
    """Run a shell command and return the result"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result

def main():
    print("🚀 Starting YOLO Tooth Detection Training Pipeline")
    
    # Step 1: Install ultralytics
    print("\n📦 Installing ultralytics...")
    run_command("pip install ultralytics")
    
    # Step 2: Check GPU availability
    print("\n🔍 Checking GPU availability...")
    run_command("nvidia-smi")
    
    # Step 3: Import required libraries
    print("\n📚 Importing libraries...")
    try:
        import torch
        from ultralytics import YOLO
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for training")
    except ImportError as e:
        print(f"Error importing libraries: {e}")
        return
    
    # Step 4: Dataset preparation
    print("\n📁 Preparing dataset...")
    
    # Check if dataset already exists
    if os.path.exists("dataset"):
        print("Dataset already exists, skipping preparation...")
    else:
        # Create dataset structure
        images_dir = "ToothNumber_TaskDataset/images"
        labels_dir = "ToothNumber_TaskDataset/labels"
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print("Error: ToothNumber_TaskDataset/images or labels directory not found!")
            print("Please ensure the dataset is properly organized.")
            return
        
        # Create train/val split
        train_ratio = 0.8
        out_dir = "dataset"
        
        # Create folders
        for split in ['train', 'val']:
            os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
        
        # Get all image filenames
        images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        def move_files(img_list, split):
            for img in img_list:
                base = os.path.splitext(img)[0]
                img_src = os.path.join(images_dir, img)
                lbl_src = os.path.join(labels_dir, base + '.txt')
                img_dst = os.path.join(out_dir, 'images', split, img)
                lbl_dst = os.path.join(out_dir, 'labels', split, base + '.txt')
                if os.path.exists(img_src) and os.path.exists(lbl_src):
                    shutil.copy2(img_src, img_dst)
                    shutil.copy2(lbl_src, lbl_dst)
        
        move_files(train_images, 'train')
        move_files(val_images, 'val')
        
        print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
        
        # Create YOLO dataset structure
        base_path = "ToothNumber_TaskDataset"
        os.makedirs(base_path, exist_ok=True)
        
        # Create subdirectories for YOLO format
        for split in ['train', 'val']:
            os.makedirs(f'{base_path}/{split}/images', exist_ok=True)
            os.makedirs(f'{base_path}/{split}/labels', exist_ok=True)
        
        # Copy training images and labels
        for img in os.listdir(os.path.join(out_dir, 'images', 'train')):
            shutil.copy2(os.path.join(out_dir, 'images', 'train', img), f'{base_path}/train/images/{img}')
        
        for lbl in os.listdir(os.path.join(out_dir, 'labels', 'train')):
            shutil.copy2(os.path.join(out_dir, 'labels', 'train', lbl), f'{base_path}/train/labels/{lbl}')
        
        # Copy validation images and labels
        for img in os.listdir(os.path.join(out_dir, 'images', 'val')):
            shutil.copy2(os.path.join(out_dir, 'images', 'val', img), f'{base_path}/val/images/{img}')
        
        for lbl in os.listdir(os.path.join(out_dir, 'labels', 'val')):
            shutil.copy2(os.path.join(out_dir, 'labels', 'val', lbl), f'{base_path}/val/labels/{lbl}')
        
        # Copy data.yaml if it exists
        if os.path.exists('data.yaml'):
            shutil.copy2('data.yaml', f'{base_path}/data.yaml')
        
        print("✅ YOLO dataset prepared successfully at:", base_path)
    
    # Step 5: Train YOLO model
    print("\n🤖 Training YOLO model...")
    
    # Check if data.yaml exists
    data_yaml_path = "dataset/data.yaml"
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found!")
        print("Please ensure the data.yaml file is properly configured.")
        return
    
    try:
        # Load pretrained YOLOv8 model
        model = YOLO('yolov8n.pt')  # Use yolov8n.pt for faster training
        
        # Train the model
        print("Starting training...")
        model.train(
            data=data_yaml_path,
            epochs=50,        # Start with 50 epochs
            imgsz=640,        # Image size
            batch=8,          # Reduced batch size for CPU
            workers=2,
            patience=10,      # Early stopping
            device='cpu'      # Force CPU training
        )
        
        print("✅ Training completed successfully!")
        
        # Step 6: Validate model
        print("\n📊 Validating model...")
        metrics = model.val()
        print("Validation metrics:")
        print(metrics)
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
