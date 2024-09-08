import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
import cv2
from ultralytics import YOLO

model = YOLO('./license_plate_model.pt')

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recognize_number_plate(image):
    """Process the image to extract number plate text."""
    
    image = np.array(image)
    
#     image_rgb = image.convert('RGB')
#     image_array = np.array(image)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image_array=np.array(image)
    
    results = model.predict(image_array, device='cpu')
    
    print(results)
    
    
    for result in results:
        for box in result.boxes:
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # crop img
            roi = image[y1:y2, x1:x2]

        # Perform OCR on the cropped image
            text = pytesseract.image_to_string(roi, config='--psm 6')
    
    return text.strip()

def main():
    st.title("Number Plate Recognition")

    st.write("Upload an image of a vehicle to recognize the number plate.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Perform number plate recognition
        st.write("Recognizing number plate...")
        number_plate = recognize_number_plate(image)
        st.write("Number Plate: ", number_plate)

if __name__ == "__main__":
    main()
