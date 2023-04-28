## PDE4434 INTELLIGENT SENSING FOR ROBOTICS


# UNO CARD RECOGNITION



## Objectives

To develop a program that can identify cards from the UNO game. Program should be able to accept input from either a file or a camera. The program should be designed to recognize one card at a time from the input image or frame.


## Steps

Navigate the folder where python file located


![command prompt 1](https://user-images.githubusercontent.com/117764288/235130156-9b672e73-0365-4e75-be58-77a66a2822cc.JPG)



Type "Python "file name.py" and press enter.


![command prompt 2](https://user-images.githubusercontent.com/117764288/235139091-1ebce26c-1d38-43fe-894c-97ee0e1bd421.JPG)

Program will run and live video open for 10 seconds, place the card infront of the camera.

It will process and display  color and number/type in window "Result"

  
    
## Working


- Import required libraries.
- Use the camera to show live video for ten seconds.
- Capture an image after ten seconds and save it as a JPEG file.
- Load the saved image and resize it.
- Process the image to extract the contour of a UNO card by converting it to gray scale, smoothing it, thresholding it, and applying edge detection.
- Find the contours in the image.
- Select the contour with the maximum area, draw a bounding rectangle around it, and crop the UNO card from the original image using the bounding rectangle. 
- Save the cropped UNO card image.
- Load a pre-trained CNN model from a file.
- Set the directory where cropped UNO card image are located.
- Loop through each image in the directory, load, resize, and display it in a new window.
- Convert the image to a numpy array and expand dimensions.
- Use the pre-trained model to predict the type of UNO card in the image.
- Display the predicted card color and number/type in window.





## Analysis
  
  

## Dependencies
 #To load a trained model
```bash
  from tensorflow.keras.models import load_model 
```
   #To interact with the file system
  
```bash
import os
```
  #To display images
```bash   
import matplotlib.pyplot as plt

```
 #To preprocess images
 
```bash   
from tensorflow.keras.preprocessing import image

```
#To measure time
  
```bash  
import time 
```
 #For numerical operations  
```bash  
import numpy as np 
```
 #For image processing and computer vision
 
```bash  
import cv2  
```



  
## Simulation



## Dataset 




## Constraints


## Future improvement


## demonstration Link 












## Reference

Following reference help me for this course work.




















## Conclusion

