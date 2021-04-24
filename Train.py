# set the matplotlib backend so figures can be saved in the background
import keras
import matplotlib
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report

from SimpleVGGNet import SimpleVGG
from EAMNet import EAMNET

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

dataset="./dataset/"
EPOCHS =100
INIT_LR = 0.01
BS = 64
IMAGE_DIMS = (64, 64, 1)
classnum=2
# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(dataset)))
# print(imagePaths)
random.seed(10010)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
    print(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
# scale the raw pixel intensities to the range [0, 1]
print(labels)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))

# 数据集切分
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

# 转换标签为one-hot encoding格式
lb = LabelBinarizer()
print(lb)
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

#trainY = to_categorical(trainY)
#testY = to_categorical(testY)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
# initialize the model
print("[INFO] compiling model...")
model = SimpleVGG.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=classnum)
# model=load_model('./model/best0428ep150.h5')
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
# train the network
filepath="./model/best0428ep150.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')

H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
   callbacks=[reduce_lr,checkpoint],
    epochs=EPOCHS, verbose=1)
# save the model to disk
model.save('./model/best0428ep150.h5')
# plot the training loss and accuracy
# 测试
print("------测试网络------")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("./model/best0428ep150.png")
f = open("./model/best0428ep150.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()
