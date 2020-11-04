import numpy as np
from keras.utils import np_utils
import os, sys

#Declare file path 
data_path='/Mask Images'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]


#Initalize arrays to store for loop results in
label_dict=dict(zip(categories,labels))
data = []
target = []


#Iterates over each image, resizes to 100x100, converts to greyscale, and stores in arrays
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        data.append(resized)
        target.append(label_dict[category])


#Divide by 255 so range of color values is on a 0.0-1.0 scale 
data=np.array(data)/255.0
#Reshape to 4 dimensions for CNN usage
data=np.reshape(data,(data.shape[0],100,100,1))
#Convert array to categorical
target=np.array(target)
target=np_utils.to_categorical(target)


#Save as Numpy Arrays
np.save('data',data)
np.save('target',target)
