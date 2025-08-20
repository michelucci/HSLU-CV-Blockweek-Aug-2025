#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


# In[2]:


def detect_and_predict_mask(frame, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    
    face = frame
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))


    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    preds = maskNet.predict(face)

    return np.argmin(preds)


# In[4]:


print("[INFO] loading face mask detector model...")
maskNet = load_model('model_mask_detection.h5')
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

color = (0, 255, 0)
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    preds = detect_and_predict_mask(frame, maskNet)
    label = 'With Mask' if (preds == 1) else 'Without Mask'
    cv2.putText(frame, label, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


# In[ ]:




