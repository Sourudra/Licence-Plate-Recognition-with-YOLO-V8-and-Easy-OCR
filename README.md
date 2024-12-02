# **Licence-Plate-Recognition-with-YOLO-V8-and-Easy-OCR**

## **Project Overview**  
This project integrates **YOLOv8** for license plate detection and **EasyOCR** for optical character recognition (OCR) to read the detected license plate numbers. The combination allows both the detection of plates in images or videos and the extraction of plate numbers in real-time.

---

## **Why YOLOv8 and EasyOCR?**  
**YOLOv8** is a state-of-the-art object detection model known for its speed and accuracy, making it ideal for real-time license plate detection. **EasyOCR**, on the other hand, specializes in text recognition, providing reliable results for reading the alphanumeric characters on license plates.

---

## **Project Details**  
The system uses **YOLOv8** to detect license plates in images or videos. Once a plate is detected, **EasyOCR** extracts the text from the plate. The results, including detected plate numbers, coordinates, and timestamps, are saved in a **CSV** file for further analysis.

---

## **Training Strategy**  
- **YOLOv8n Model:** A pre-trained YOLOv8 model is fine-tuned with a dataset of license plates to improve detection accuracy.  
- **EasyOCR:** OCR is applied on the cropped license plate regions to read the numbers.  
- **Training Epochs:** The model was trained for 100 epochs to ensure optimal performance.

---

## **Key Features**  
- **Real-Time Detection & Recognition:** Capable of detecting and recognizing license plates from live or recorded video feeds.  
- **High Accuracy:** The combination of YOLOv8 for detection and EasyOCR for recognition ensures reliable performance.  
- **CSV Logging:** Plate numbers, coordinates, and timestamps are saved for easy access and further processing.  

---

## **Future Directions**  
- **Dataset Expansion:** Increasing the variety of license plate samples to improve model generalization across regions.  
- **Hyperparameter Tuning:** Further optimization of both models for enhanced accuracy and performance.  
- **Model Optimization:** Exploring methods for reducing inference time and adapting the system for edge devices.  

---
## **How to train the model?**  
For detailed information on how I created the model and further insights into the training process, visit my [other repository here](https://github.com/Sourudra/Fine-tuning-YOLOv8-for-Licence-Plate-Detection).
---
