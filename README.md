## PDE4434 INTELLIGENT SENSING FOR ROBOTICS


# UNO CARD RECOGNITION



## Objectives

To develop a program that can identify cards from the UNO game. Program should be able to accept input from either a file or a camera. The program should be designed to recognize one card at a time from the input image or frame.


## Steps


![command prompt 1](https://user-images.githubusercontent.com/117764288/235130156-9b672e73-0365-4e75-be58-77a66a2822cc.JPG)



![Uploading command prompt 2.JPG…]()

  
    
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
- Display the predicted card color and number/type.





## Analysis
  
  

## Dependencies

```bash
  from tensorflow.keras.models import load_model 
```
```bash
from tensorflow.keras.models import load_model
```
   #To load a trained model
```bash
import os
```
   #To interact with the file system
```bash   
import matplotlib.pyplot as plt

```
   #To display images
```bash   
from tensorflow.keras.preprocessing import image

```

  #To preprocess images
```bash  
import time 
```
  #To measure time
```bash  
import numpy as np 
```
  #For numerical operations
```bash  
import cv2  
```

#For image processing and computer vision

  
## Simulation



## Dataset 




## Constraints


## Future improvement


## demonstration Link 












## Reference

Following reference help me for this course work.




















## Conclusion

