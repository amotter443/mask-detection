import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2

#Pull in image, crop to square & pre-process
shaq = cv2.imread('/shaq_mask.jpg')  
shaq = shaq[1:621, 330:950]
shaq=cv2.resize(shaq,(100,100))
shaq=cv2.cvtColor(shaq,cv2.COLOR_BGR2GRAY)

#Load pre-trained model
model = load_model('/mask_nn.model')

#Apply model to Shaq image
pred = model.predict(shaq.reshape(1,100,100,1))

#Output image and prediction
print(pred.argmax())
plt.imshow(shaq,cmap='gray');