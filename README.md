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


Live camera captured the image and saved in to located folder


![test](https://user-images.githubusercontent.com/117764288/235215245-78f5ca78-7aa7-4a94-8d97-aebbcf398392.jpeg)


Cropped image and saved into folder called "images" .


![Test1](https://user-images.githubusercontent.com/117764288/235215313-d664306d-d76c-4013-8322-ccee69e6c8d4.jpeg)




It will process and display  color and number/type in window "Result"


  
![output](https://user-images.githubusercontent.com/117764288/235216180-86088767-79e6-4d18-a4b6-c4e3731c9393.JPG)



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

In this project using simple cnn model having sequence of layers that will process our images. The first layer is a convolutional layer that will learn important features of our images. We then add a second convolutional layer, a max pooling layer to down sample the features, and a dropout layer to prevent overfitting. We then flatten our data and add two fully connected layers to output our predictions.



  

![Capturedd](https://user-images.githubusercontent.com/117764288/235227239-a071efab-e568-4800-96ef-79ca0e1e7337.JPG)








During testing on a dataset, the model achieved an accuracy of 99.12% with a loss of 0.0233.


![image](https://user-images.githubusercontent.com/117764288/235227759-97305af0-9861-4d43-bc46-1eb42a21e7ef.png)





## Dataset 

My dataset have 15 classes, its helps to identify the number/action using CNN model.

Each classes include cropped images, these images are trained to detect the card number/actions.

```bash

['1', '2', '3', '4', '5', '6', '7', '8', '9', 'Draw 2', 'NO CARD DETECTED', 'Reverese', 'Wild card', 'Wild card draw 4', 'skip']

```

Datasample include,example 4 colors have number 1, so we created the cropped data set according to our program.


![data sample](https://user-images.githubusercontent.com/117764288/235218374-16d7d0fd-3c00-4e40-892e-bb1899700976.JPG)


## Condition

- Images are captured under daylight and room light conditions ,still possibility to give the wrong output, for better result , do the trials in room lighting with black background.


## Future improvement

1.Train the model with large dataset having different condition images to train model get better result.

2.identify the card detection automatically using computer vision methods.











## Conclusion






## Demonstration Link 


Video 1: https://www.youtube.com/watch?v=3lpa2Ujsyjs

Video 2 :https://www.youtube.com/watch?v=0wO6pnyh2es




## Reference

Following reference help me for this course work.

https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html<br>
https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html<br>
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_hierarchy/py_contours_hierarchy.html
https://en.wikipedia.org/wiki/HSL_and_HSV





