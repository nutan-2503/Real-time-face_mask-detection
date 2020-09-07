"""
@author: Nutan Hotwani
"""

# Importing libraries
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt 

# Loading model architecture and weights
def load_model():
    with open('E:/face_mask/json_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.summary()
    loaded_model.load_weights('weights.h5')
    print("Model loaded...")
    return loaded_model

# detect face mask:
def detect_face_mask(img):
      
    global loaded_model    
    label_dict = {0:'without_mask', 1:'mask'}
    color_dict = {0:(0, 0, 255), 1:(0, 255, 0)}
    
    dims = cascade.detectMultiScale(img)
    
    for (x,y,w,h) in dims:
        
        roi = img[y:y+h , x:x+w]
        #roi = cv2.cvtColor(roi, 1) #cv2.BGR2GRAY
        roi = cv2.resize(roi , dsize=(224, 224))
        normalized = roi/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=loaded_model.predict(reshaped)
        #print(Class.shape)
                
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, label_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    return img


# Excution:
# Loading video:
video_path = "video.mp4" # video
harr_path = "Haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(harr_path)

loaded_model = load_model()

# Real-time processing
cam = cv2.VideoCapture(video_path)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('Result.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

i=0
while i<2000:
    _, frame = cam.read()
    frame = detect_face_mask(cv2.flip(frame , 1, 1))
    writer.write(frame)
    i+=1

cam.release()
writer.release()
cv2.destroyAllWindows()
