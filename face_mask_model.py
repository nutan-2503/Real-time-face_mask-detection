"""
@author: Nutan Hotwani
"""

## Importing Libraries ##
import numpy as np
import pandas as pd
import glob
from google.colab import drive
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Activation, Dropout
from keras.optimizers import Adam, RMSprop

# Function for loading images from dest
def load_images(data_path, dest):
  image=[]
  img_width=224
  img_height=224
  for file in glob.glob(data_path+dest):
    image.append(cv2.resize(cv2.imread(file), (img_width, img_height), interpolation=cv2.INTER_CUBIC))
  return image

# Creating dataset (75% training data)
def create_data(image_mask, image_no_mask):
    image=[]
    image=image_mask+image_no_mask
    y=[1]*len(image_mask)+[0]*len(image_no_mask)
    data=pd.DataFrame({"Image":image, "Label":y})
    data=shuffle(data) # data: contains total dataset with images in array and labels as (0: no_mask, 1:mask)
    train, test=train_test_split(data, test_size=0.25, random_state=7, shuffle=True)
    X_train=train.Image
    y_train=train.Label
    X_test=test.Image
    y_test=test.Label
    y_train=to_categorical(y_train, 2)
    y_test=to_categorical(y_test, 2)
    X_test=np.array(X_test)/255.0 # normalization 
    X_train=np.array(X_train)/255.0
    X_train=np.stack(X_train, axis=0) # creating a 4-tuple dataset 
    X_test=np.stack(X_test, axis=0)
    return X_train, y_train, X_test, y_test
    
# Designing CNN model to be used
def cnn_model():
    model=Sequential()

    model.add(Conv2D(224, (3, 3), padding='same', input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(150, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.20))
    
    model.add(Flatten())
    model.add(Dropout(0.25))
    
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
    
# Plotting accuracy and loss graph
def plot_graph(hist):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'test'])
    
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'test'])
    
    plt.show()

# Execution:
data_path = "E:/face_mask/"
image_mask = load_images(data_path, dest="with_mask/*.*")
print(len(image_mask)) # Size of masked dataset

image_no_mask = load_images(data_path, dest="without_mask/*.*")
print(len(image_no_mask))

X_train, y_train, X_test, y_test = create_data(image_mask, image_no_mask)

model = cnn_model()

# training on X_train for 20 epochs with validation split of 0.2
hist=model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.2)

# Plotting accuarcy and loss 
plot_graph(hist)

# Test data accuracy
_, test_acc=model.evaluate(X_test, y_test, verbose=0)
print(test_acc)

# saving model for real life detection
# serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('model.json', 'w') as json_file:
    json_file.write(json_model)
#saving the weights of the model
model.save_weights('weights.h5')





    

    

