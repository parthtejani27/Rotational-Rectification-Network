#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import shutil

source_dir = 'F:/dataset'  # Path to the dataset folder
destination_dir = 'E:/Updated Dataset'  # Path to the destination folder

sets = ['set00','set01','set02','set03','set04','set05']

# Create train and test directories in the destination folder
train_dir = os.path.join(destination_dir, 'train')
test_dir = os.path.join(destination_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

img_dest_dir = os.path.join(train_dir,'images')
annot_dest_dir = os.path.join(train_dir,'annotations')
os.makedirs(img_dest_dir, exist_ok=True)
os.makedirs(annot_dest_dir, exist_ok=True)


for set_name in sets:
    set_dir = os.path.join(source_dir, set_name)
    videos = [f for f in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, f))]
    for video_name in videos:
        video_dir = os.path.join(set_dir, video_name)
        img_source_dir = os.path.join(video_dir, 'images')
        annot_source_dir = os.path.join(video_dir, 'annotations')
        
        for img_file in os.listdir(img_source_dir):
            img_source_path = os.path.join(img_source_dir, img_file)
            img_dest_path = os.path.join(img_dest_dir, set_name + '_' + video_name + '_' + img_file)
            shutil.copy(img_source_path, img_dest_path)
            annot_dest_path = os.path.join(annot_dest_dir, set_name + '_' + video_name + '_' + img_file[:-4]+'.json')
            shutil.copy(annot_source_path, annot_dest_path)
            
        for annot_file in os.listdir(annot_source_dir):
            annot_source_path = os.path.join(annot_source_dir, annot_file)
            annot_dest_path = os.path.join(annot_dest_dir, set_name + '_' + video_name + '_' + annot_file)
            shutil.copy(annot_source_path, annot_dest_path)


# In[11]:


from PIL import Image

# Rotate function
def rotate_image(image, angle):
    rotated_image = image.rotate(angle)
    rotated_image = np.array(rotated_image)
    return rotated_image


# In[20]:


import pandas as pd
import os
import numpy as np

training_Path_To_Rotate = "E:/Updated Dataset/train/images"
testing_Path_To_Rotate = "E:/Updated Dataset/test/images"

rotated_training_path = "E:/Updated Dataset/train/rotated_images"
rotated_testing_path = "E:/Updated Dataset/test/rotated_images"

rotated_train_json_angle_path = "E:/Updated Dataset/train/angle.json"
rotated_test_json_angle_path = "E:/Updated Dataset/test/angle.json"

os.makedirs(rotated_training_path, exist_ok=True)
os.makedirs(rotated_testing_path, exist_ok=True)

rotated_training_angles_dict = []
rotated_testing_angles_dict = []

for filename in os.listdir(training_Path_To_Rotate):
    if filename.endswith('.jpg'):
        image_path = os.path.join(training_Path_To_Rotate, filename)
        angle = np.random.randint(-90, 91)
        rotated_image = rotate_image(np.array(Image.open(image_path).convert('L')), angle)
        
        rotated_filename = os.path.splitext(filename)[0] + '_rotated_' + str(angle) + '.jpg'
        rotated_image_path = os.path.join(rotated_training_path, rotated_filename)
        Image.fromarray(rotated_image.astype('uint8')).save(rotated_image_path)
        
        rotated_training_angles_dict.append([rotated_filename,angle])
        

    
for filename in os.listdir(testing_Path_To_Rotate):
    if filename.endswith('.jpg'):
        image_path = os.path.join(testing_Path_To_Rotate, filename)
        angle = np.random.randint(-90, 91)
        rotated_image = rotate_image(np.array(Image.open(image_path).convert('L')), angle)
        
        rotated_filename = os.path.splitext(filename)[0] + '_rotated_' + str(angle) + '.jpg'
        rotated_image_path = os.path.join(rotated_testing_path, rotated_filename)
        Image.fromarray(rotated_image.astype('uint8')).save(rotated_image_path)
        
        rotated_testing_angles_dict.append([rotated_filename,angle])
        


# In[21]:


rotated_training_angles_dict=pd.DataFrame(rotated_training_angles_dict,columns=["Filename","angle"])
rotated_testing_angles_dict=pd.DataFrame(rotated_testing_angles_dict,columns=["Filename","angle"])

rotated_training_angles_dict.to_csv(rotated_train_json_angle_path)
rotated_testing_angles_dict.to_csv(rotated_test_json_angle_path)


# In[18]:



import cv2
import json

# Load JSON data containing annotations
json_path ="F:/dataset/set00/V000/annotations/I00297.json"

with open(json_path, 'r') as json_file:
    json_data = json.load(json_file)

# Read the image file
image_path = "E:/Caltech Dataset With Annotation and rotation/train/images/set00_V000_I00297.jpg"
image = cv2.imread(image_path)

# Iterate through each annotation
for annotation in json_data:
    # Extract annotation details (e.g., bounding box coordinates)
    x, y, width, height = map(int, annotation['pos'])

    # Draw bounding box rectangle on the image
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Display the image with annotations
cv2.imshow('Image with Annotations', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[16]:


get_ipython().system('pip install --upgrade protobuf')
get_ipython().system('pip install protobuf==3.19.0')
get_ipython().system('pip install tensorflow')
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Flatten, concatenate, ZeroPadding2D

def create_model(input_shape):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional Layer 1
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu')(inputs)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(norm1)

    # Convolutional Layer 2
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Convolutional Layer 3
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu',padding='same')(pool2)

    # Convolutional Layer 4
    conv4 = Conv2D(256, kernel_size=(3, 3), activation='relu',padding="same")(conv3)

    # Convolutional Layer 5
    conv5 = Conv2D(256, kernel_size=(3, 3), activation='relu',padding='same')(conv4)

    # # Adjust spatial dimensions of conv3 and conv4
    # conv3_pad = ZeroPadding2D(padding=((0, 0), (0, 0)))(conv3)  # Pad conv3 to match conv5
    # conv4_pad = ZeroPadding2D(padding=((0, 0), (0, 0)))(conv4)  # Pad conv4 to match conv5

    # Concatenate conv3, conv4, and conv5
    concat = concatenate([conv3, conv4, conv5])

    # Global Pooling
    gp_pool = GlobalMaxPooling2D()(concat)

    # Flatten
    flatten = Flatten()(gp_pool)

    # Output layer (Theta)
    theta = tf.keras.layers.Dense(units=1, activation='linear')(flatten)

    # Create the model
    model = Model(inputs=inputs, outputs=theta)

    return model

# Define the input shape
input_shape = (480, 640, 3)  # 480x640 RGB images

# Create the model
model = create_model(input_shape)

# Print the model summary
model.summary()


# In[ ]:




