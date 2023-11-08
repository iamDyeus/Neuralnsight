import cv2
from pathlib import Path
import os
import numpy as np 
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard



# Step 1 - Data Extraction
def relative_to_Dataset(foldername)->str:
    """
    Returns the path of the foldername in the Dataset folder
    """
    return str(Path.cwd() / 'Dataset' / foldername)

def list_of_images(foldername)->list:
    """
    Returns the list of images in the foldername
    """
    folder = relative_to_Dataset(foldername)
    images = os.listdir(folder)
    return images


# DATA PREPROCESSING
dataset = list()
labels = list()
INPUT_SIZE = (64,64)


# Resizing Images
## Resizing "NOT BRAIN TUMOR" Images
for i,image_name in enumerate(list_of_images('no')):
    if(image_name.endswith('.jpg')):
        image = cv2.imread(relative_to_Dataset('no') + '/' + image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize(INPUT_SIZE)
        dataset.append(np.array(image))
        labels.append(0)

## Resizing "BRAIN TUMOR" Images
for i,image_name in enumerate(list_of_images('yes')):
    if(image_name.endswith('.jpg')):
        image = cv2.imread(relative_to_Dataset('yes') + '/' + image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize(INPUT_SIZE)
        dataset.append(np.array(image))
        labels.append(1)


# print("Dataset and Labels are ready for training")
# print("Dataset Shape: ",np.array(dataset).shape)
# print("Labels Shape: ",np.array(labels).shape)
# print("Dataset Type: ",type(dataset))
# print("Labels Type: ",type(labels))
# print("Dataset Length: ",len(dataset))
# print("Labels Length: ",len(labels))
# print("Dataset : ",dataset)
# print("Labels : ",labels)


# Converting Lists to Numpy Arrays
dataset = np.array(dataset)
labels = np.array(labels)


# Splitting the Dataset into Training and Testing
X_train,X_test,y_train,y_test = train_test_split(dataset,labels,test_size=0.2,random_state=0)

# Reshaping the Training and Testing Data
X_train = normalize(X_train,axis=1)
X_test = normalize(X_test,axis=1)

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)


# Step 2 - Building the Model

# Initialising the CNN
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(INPUT_SIZE[0],INPUT_SIZE[1],3),activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

no_of_conv_layers = 2
for i in range(no_of_conv_layers):
    model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=16,verbose=True,epochs=10,validation_data=(X_test,y_test),shuffle=True)

model.save('BrainTumorModel_10epochs_Categorical.h5')

