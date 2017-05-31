import csv
import numpy as np
import cv2
import sklearn
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation, ELU 
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_data(data):

    images, measurements = [], []
    
    # Steering correction used for left and right images
    correction = 0.25
    
    with open(data, 'r') as f:
        
        reader = csv.reader(f)
        
        # skip header
        next(reader, None)
        
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            center_img = './data/IMG/' + center_img.strip().split('/')[-1]
            left_img = './data/IMG/' + left_img.strip().split('/')[-1]
            right_img = './data/IMG/' + right_img.strip().split('/')[-1]

            images.append([center_img, left_img, right_img])
            measurements.append([angle, angle + correction, angle - correction])

    return images, measurements


def resize(image):
    """
    Returns resized image to feed the network.
    """
    return cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)


def normalize(image):
    """
    Returns a normalized image with feature values from -1.0 to 1.0.
    """
    return image / 255.0 - 0.5


def crop_image(image):
    """
    Returns an image cropped 40 pixels from top and 20 pixels from bottom.
    """
    return image[40:-20,:]


def random_brightness(image):
    """
    Returns an image with a random degree of brightness.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def preprocess(image):
    """
    Preprocess the images from the car simulator.
    """
    image = random_brightness(image)
    image = crop_image(image)
    image = resize(image)
    return image


def generator(X_train, y_train, batch_size = 64):
    """
    Return two numpy arrays containing images and their associated steering angles.
    :param X_train: A list of image names to be read in from data directory.
    :param y_train: A list of steering angles associated with each image.
    :param batch_size: The size of the numpy arrays to be return on each pass.
    """
    images = np.zeros((batch_size, 66, 200, 3), dtype = np.float32)
    steerings = np.zeros((batch_size,), dtype = np.float32)
    
    while True:
        
        straights = 0
        
        for i in range(batch_size):
            
            # Randomly select an index of sample
            sample_index = random.randrange(len(X_train))
            
            # Randomly select a camera to use: Left -> 1, Right -> 2 or Center -> 0
            image_index = random.randrange(len(X_train[0]))
            
            # Read the corresponding steering angle
            steering = y_train[sample_index][image_index]
            
            # Limit angles of less than absolute value of .1 to no more than 1/2 of data
            # to reduce bias of car driving straight
            if abs(steering) < .1:
                straights += 1
            if straights > (batch_size * .5):
                while abs(y_train[sample_index][image_index]) < .1:
                    sample_index = random.randrange(len(X_train))
            
            # Read image in from directory, process, and convert to numpy array
            #  print(str(X_train[sample_index][image_index]))
            image = cv2.imread(str(X_train[sample_index][image_index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess(image)
            image = np.array(image, dtype=np.float32)
            
            # Flip image and apply opposite angle 50% of the time
            if random.randrange(2) == 1:
                image = cv2.flip(image, 1)
                steering = steering * -1.0
            images[i] = image
            steerings[i] = steering

        yield images, steerings


def build_model():
    
    model = Sequential()

    # Normalize layer
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))

    # Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | elu activation 
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)))
    
    # Dropout with drop probability of .1 (keep probability of .9)
    model.add(Dropout(.1))
    
    # Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | elu activation
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)))
    
    # Dropout with drop probability of .2 (keep probability of .8)
    model.add(Dropout(.2))
    
    # Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | elu activation
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)))
    
    # Dropout with drop probability of .2 (keep probability of .8)
    model.add(Dropout(.2))
    
    # Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | elu activation
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)))
    
    # Dropout with drop probability of .2 (keep probability of .8)
    model.add(Dropout(.2))
    
    # Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | elu activation
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)))
    
    # Flatten
    model.add(Flatten())
    
    # Dropout with drop probability of .3 (keep probability of .7)
    model.add(Dropout(.3))
    
    # Fully-connected layer 1 | 100 neurons | elu activation
    model.add(Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)))
    
    # Dropout with drop probability of .5
    model.add(Dropout(.5))
    
    # Fully-connected layer 2 | 50 neurons | elu activation
    model.add(Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)))
    
    # Dropout with drop probability of .5
    model.add(Dropout(.5))
    
    # Fully-connected layer 3 | 10 neurons | elu activation
    model.add(Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)))
    
    # Dropout with drop probability of .5
    model.add(Dropout(.5))
    
    # Output
    model.add(Dense(1, activation='linear', init='he_normal'))
        
    model.compile(optimizer='adam', loss='mse')
    
    return model   


if  __name__ == "__main__":

    # Read the log file
    X_train, y_train = read_data('data/driving_log.csv')
    
    # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    
    # Split the data for train and validation set
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.15)

    # Build model
    model = build_model()
    
    model.summary()
    
    # For Keras 1.2.x
    history_object = model.fit_generator(generator(X_train, y_train), samples_per_epoch = 24000,
                                         validation_data = generator(X_validation, y_validation), 
                                         nb_val_samples = 1024, nb_epoch = 30, verbose = 1)
    
    # For Keras 2.0 and above
    #history_object = model.fit_generator(generator(X_train, y_train), steps_per_epoch = 750,
    #                                     epochs = 5, verbose = 1,
    #                                     validation_data = generator(X_validation, y_validation),
    #                                     validation_steps = 32)
    
    # Save the model
    model.save('model.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')

    plt.tight_layout()
    plt.savefig('loss.png', dpi=200)
    #plt.show()