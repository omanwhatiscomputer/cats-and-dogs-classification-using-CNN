# -*- coding: utf-8 -*-

# Part-1 Data Preprocessing-Done Manually

# Part-2 Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D #To add the convolution layers
from keras.layers import MaxPooling2D # To add the pooling layers
from keras.layers import Flatten # Flatten our pooled feature maps
from keras.layers import Dense # To add all the fully connected layers (ANN)

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution --Creating the feature maps using feature detectors
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
# 32(number of feature detectors) is the convention, 3x3 filter size; input_shape- (imageSize x imageSize) and no of channels=3

# Step 2 - Pooling --Reducing the size of our feature map and add spacial features to our feature maps
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3 - Flattening -- Flatten the reduced feature map
# the max numbers in each pooled feature maps helps retain the information of the spacial structure
classifier.add(Flatten())


# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')

# Part 2 - Fitting the CNN to the image- Image augtentation process --prevents overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),# input_shape/image size as expected by our CNN
        batch_size=32, # per iteration of our SGD
        class_mode='binary')# number of categories in our dependent variable

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data= test_set, #evaluate the performance
        validation_steps=2000)

# Part-3 Making new Predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/car_or_dog_1.jpg', target_size= (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)