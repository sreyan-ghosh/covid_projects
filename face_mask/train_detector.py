# we will use shell commands to train the model and perform actions
# the necessary commands will be mentioned wherever reqd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# creating the argument parser and parsing the shell commands
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the input dataset')
ap.add_argument('-p', '--plot', type=str, default='plot.png', help='path to output loss/accuracy plot')
ap.add_argument('-m', '--model', type=str, help='path to output model')
args = vars(ap.parse_args())

# initializing the global variables
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# run the program using python3 train_detector.py -d /path/to/dataset_dir
# (not necessary to run, I have already run it and the mask_detector.model file is already created.)

# initialise the input images in their directory
print('loading images...')
image_paths = list(paths.list_images(args['dataset']))
data = []
labels = []

# looping over the image paths
for image_path in image_paths:
    # extract the class label from the filename, but it must follow the current directory structure
    label = image_path.split(os.path.sep)[-2]

    # load and prerocess the image
    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # add the image data and labels to lists
    data.append(image)
    labels.append(label)

# update the data and labels list to numpy arrays
data = np.array(data, dtype='float32')
labels = np.array(labels)

# perform OHE on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# split the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, stratify=labels)

# construct a data generator for image augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# load the MobileNetV2() model in the headless mode
base_model = MobileNetV2(weights='imagenet', include_top=False, 
            input_tensor=keras.layers.Input(shape=(224,224,3)))

# constructing the head of the model
head_model = base_model.output
head_model = keras.layers.AveragePooling2D(pool_size=(7,7))(head_model)
head_model = keras.layers.Flatten(name='flatten')(head_model)
head_model = keras.layers.Dense(128, activation='relu')(head_model)
head_model = keras.layers.Dropout(0.5)(head_model)
head_model = keras.layers.Dense(2, activation='softmax')(head_model)

# creating the final trainable model 
model = Model(inputs=base_model.input, outputs=head_model)

# making the layers of the base_model untrainable
for layer in base_model.layers:
    layer.trainable = False

# compiling our model
print('compiling model...')
opt = keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the head of the network
print('training model...')
history = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX)//BS,
    validation_data=(testX, testY),
    validation_steps=len(testX)//BS,
    epochs=EPOCHS
)

# make predictions on the testing network
print('evaluating model...')
pred_idxs = model.predict(testX, batch_size=BS)

# we need to find out the predicted values for each column
pred_idxs = np.argmax(pred_idxs, axis=1)

# displaying the classification report
print(classification_report(testY.argmax(axis=1), pred_idxs, target_names=lb.classes_))

# serialize the model to disk
print('saving model...')
model.save(args['model'], save_format='h5')

# plot the training and testing loss and accuracy
n = EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,n), history.history['loss'], label='Training Loss')
plt.plot(np.arange(0,n), history.history['val_loss'], label='Validation Loss')
plt.plot(np.arange(0,n), history.history['accuracy'], label='Training Accuracy')
plt.plot(np.arange(0,n), history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Losses and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])
