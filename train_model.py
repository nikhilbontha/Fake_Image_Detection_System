import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

real_path="dataset/real"
fake_path="dataset/fake"

data=[]
labels=[]

def load_images(path,label):

    for img in os.listdir(path):

        img_path=os.path.join(path,img)

        image=cv2.imread(img_path)

        if image is None:
            continue

        image=cv2.resize(image,(128,128))

        image=image/255.0

        data.append(image)
        labels.append(label)

load_images(real_path,0)
load_images(fake_path,1)

data=np.array(data)
labels=np.array(labels)

print("Total Images:",len(data))

X_train,X_test,y_train,y_test=train_test_split(
    data,labels,test_size=0.2,random_state=42
)

# CNN Feature Extraction + Classification

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history=model.fit(
    X_train,y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test,y_test)
)

model.save("fake_image_model.h5")

# Accuracy Graph

plt.figure()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.show()

# Loss Graph

plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.show()

# Confusion Matrix

pred=model.predict(X_test)

pred=(pred>0.5)

cm=confusion_matrix(y_test,pred)

plt.figure()

sns.heatmap(cm,annot=True,cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()