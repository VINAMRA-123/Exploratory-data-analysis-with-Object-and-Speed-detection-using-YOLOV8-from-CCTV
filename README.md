## ABSTRACT

This project focuses on the development of an intelligent object detection system designed for CCTV footage, including the detection of vehicle speed. The goal is to enhance surveillance capabilities by integrating advanced computer vision, deep learning, and data analysis techniques. The approach combines Convolutional Neural Networks (CNNs) with state-of-the-art object detection algorithms such as YOLO (You Only Look Once) and Darknet. The primary aim is to achieve real-time processing of CCTV feeds, ensuring prompt identification and response to potential security threats.

In this project, I have trained the model to recognize 20 different objects, including 'person,' 'car,' 'chair,' 'bottle,' 'potted plant,' 'bird,' 'dog,' 'sofa,' 'bicycle,' 'horse,' 'boat,' 'motorbike,' 'cat,' 'TV monitor,' 'cow,' 'sheep,' 'airplane,' 'train,' 'dining table,' and 'bus.' Following the Data Science Software Development Life Cycle (SDLC), I gathered useful data and requirements, created a pipeline to prepare the data, and trained the YOLO model in Jupyter Notebook. The final step involved detecting objects using the trained model.

Additionally, the software finds the speed of vehicles by determining the time taken to travel between two imaginary lines with the help of YOLOv8, which has more efficiency than any other YOLO version.

---

## METHODOLOGIES AND CONCEPTS

### 3.1 Theoretical Concept of YOLO

YOLO (You Only Look Once) is a real-time object detection algorithm that divides the input image into cells and predicts bounding boxes and class probabilities for objects within each cell.

### 3.2 Preparing the Dataset and Annotations for Training

1. **Object Labeling**: Used labeling tools to annotate objects within images, specifying bounding boxes.
2. **Annotation File Creation**: Generated text files with bounding box coordinates and object class IDs for each image.
3. **Organizing Directory Structure**: Structured the dataset into training and validation sets.

### 3.3 Training the YOLO Model Using Darknet

- Created a class `YOLO_Pred` to facilitate object detection using YOLO.
- Initialized the model with pre-trained weights and configured it for object detection.
- Implemented the `predictions` method to perform object detection and draw bounding boxes around detected objects.
- Used non-maximum suppression to refine and filter detections.

### 3.4 Fine-tuning the Pre-trained YOLO Model for Better Accuracy

- Prepared a custom dataset with annotated objects.
- Customized YOLO configuration for the specific number of classes.
- Initialized the model with pre-trained weights and replaced the last layer to match the custom dataset.
- Conducted fine-tuning by training the model on the custom dataset.
- Optimized learning rates and hyperparameters for better performance.

### 3.5 Testing the Trained Model on Images and Videos

- Evaluated model performance using a confusion matrix and precision-recall curves.
- Achieved precision and recall of over 70%.
- Visualized detection results on images and videos.

### 3.6 Visualizing the Results and Tuning the Parameters for Better Performance

- Displayed input and output images with detected objects.
- Iteratively tuned parameters to improve detection accuracy.

### 3.7 Speed Detection

- Created a class for tracking vehicles detected in video feeds, assigning unique IDs and updating their positions.
- Calculated vehicle speeds based on position changes over time.
- Established virtual lines on the frame to monitor vehicle movement direction.
- Implemented main script for real-time visualization of tracked objects and vehicle speeds.

### Results

- Evaluated system performance on various video clips with different traffic densities and vehicle speeds.
- Achieved accurate vehicle tracking and speed estimation with mean absolute error within acceptable range.

### Outputs

- Displayed tracked vehicles with IDs and calculated speeds in the video feed.

---

### Conclusion and Future Scope

The project successfully developed an intelligent object detection system for CCTV footage, incorporating real-time processing and vehicle speed detection. Future work could involve expanding the object classes, improving detection accuracy, and exploring additional applications in public safety and transportation security.


