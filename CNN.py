import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import sys
sys.stdout.reconfigure(encoding='utf-8')

testRatio = 0.15
ValidationRation = 0.15
path = 'brisc2025_balanced_aug/train'
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
imgDim = (512, 512, 3)

images = []
classNo = []

print(path, "contains:", class_names)
numOfClasses = len(class_names)
print("Importing data...")

for classIndex, className in enumerate(class_names):
    imageList = os.listdir(os.path.join(path, className))
    for imageFile in imageList:
        curImg = cv2.imread(os.path.join(path, className, imageFile))
        curImg = cv2.resize(curImg, (50, 50))
        images.append(curImg)
        classNo.append(classIndex)
    print(className, end=".")

print("")
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(classNo.shape)

x_train, x_test, y_train, y_test = train_test_split(
    images, classNo, test_size=testRatio, shuffle=True, stratify=classNo
)
x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, y_train, test_size=ValidationRation, shuffle=True, stratify=y_train
)

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

x_train = np.array(list(map(preprocess, x_train)))
x_test = np.array(list(map(preprocess, x_test)))
x_validation = np.array(list(map(preprocess, x_validation)))

x_train = x_train.reshape(x_train.shape[0], 50, 50, 1)
x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)
x_validation = x_validation.reshape(x_validation.shape[0], 50, 50, 1)

print("x train:", x_train.shape)
print("x test:", x_test.shape)

numOfSamples = [len(np.where(y_train == x)[0]) for x in range(numOfClasses)]
plt.figure(figsize=(8, 4))
plt.bar(range(0, numOfClasses), numOfSamples)
plt.title("Number of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Count")
plt.show()

y_train = to_categorical(y_train, numOfClasses)
y_test = to_categorical(y_test, numOfClasses)
y_validation = to_categorical(y_validation, numOfClasses)

noOfFilters = 60
sizeOfFilter1 = (5, 5)
sizeOfFilter2 = (3, 3)
sizeOfPool = (2, 2)
noOfNodes = 500
batchSize = 60
epochVal = 10

def myModel():
    model = Sequential(name="Brain_Tumor_CNN")
    model.add(Conv2D(noOfFilters, sizeOfFilter1, input_shape=(50, 50, 1), activation='relu', name='conv1'))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=sizeOfPool, name='pool1'))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu', name='conv3'))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=sizeOfPool, name='pool2'))
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(noOfNodes, activation='relu', name='dense1'))
    model.add(Dropout(0.5, name='dropout2'))
    model.add(Dense(numOfClasses, activation='softmax', name='output'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
model.build(input_shape=(None, 50, 50, 1))
model.summary()

history = model.fit(
    x_train, y_train,
    batch_size=batchSize,
    steps_per_epoch=len(x_train) // batchSize,
    epochs=epochVal,
    validation_data=(x_validation, y_validation),
    shuffle=True
)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss =", score[0])
print("Test Accuracy =", score[1])

model.save("IRM_CNN.h5")
print("✅ Model saved successfully as IRM_CNN.h5")
