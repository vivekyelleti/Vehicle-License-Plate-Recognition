# Vehicle-License-Plate-Recognition

# Project Overview

This project implements a number plate recognition system using YOLOv8 for detecting number plates in images and Tesseract OCR for extracting text from the detected plates. The project also features a simple, user-friendly web interface built with Streamlit, where users can upload images and obtain the recognized number plate text.

# Dataset
The dataset used for training and testing the YOLOv8 model is sourced from Kaggle’s Object Detection YOLO OCR dataset. It contains annotated images of vehicles with visible number plates, ensuring a wide variety of images for robust model training and testing.

Source of the dataset: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data 

# Features

YOLOv8 Object Detection: Detects vehicle number plates in real-time from images.

OCR with Tesseract: Extracts the text from the detected number plates.

Streamlit Web Interface: Provides a user-friendly interface to upload images and view recognized number plates.

Bounding Box Visualization: Displays the bounding box around the detected number plate for easy verification.
Tech Stack

YOLOv8: Object detection model for locating number plates.

Tesseract OCR: Optical Character Recognition for extracting the text.

Streamlit: Web-based user interface.

OpenCV: Image processing library.

Python: Programming language.

# How it Works

1. The user uploads an image via the Streamlit interface.
2. YOLOv8 detects the number plate and creates a bounding box around it.
3. The detected plate region is passed to Tesseract OCR for text recognition.
4. The recognized text is displayed on the screen along with the bounding box.

#Contributing

Feel free to submit issues or pull requests to improve the project.


