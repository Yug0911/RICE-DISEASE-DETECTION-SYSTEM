import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, train_dir, test_dir, img_size=(224, 224), batch_size=32):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = None
        self.num_classes = 0

    def get_data_generators(self, augmentation=True):
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.15
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.15
            )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        val_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        train_generator.class_indices

        return train_generator, val_generator, test_generator

    def get_class_distribution(self, directory):
        class_counts = {}
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len(os.listdir(class_path))
        return class_counts

    def print_dataset_info(self, train_gen, val_gen, test_gen):
        print("=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.class_names}")
        print(f"Image size: {self.img_size}")
        print(f"Batch size: {self.batch_size}")
        print("=" * 50)


def load_data():
    train_dir = "Rice Dataset/Augmented Dataset/Part-1/After Augmentation"
    test_dir = "Rice Dataset/Original Dataset"

    loader = DataLoader(train_dir, test_dir, img_size=(224, 224), batch_size=32)
    train_gen, val_gen, test_gen = loader.get_data_generators(augmentation=True)

    loader.print_dataset_info(train_gen, val_gen, test_gen)

    return train_gen, val_gen, test_gen, loader.class_names


if __name__ == "__main__":
    train_gen, val_gen, test_gen, class_names = load_data()