import csv
import cv2
import numpy as np

# read in the driving log data
lines = []
with open('./data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

for line in lines:
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    path = './data/IMG/' # fill in the path to your training IMG directory
    img_center = cv2.imread(path + line[0].split('/')[-1])
    img_left = cv2.imread(path + line[1].split('/')[-1])
    img_right = cv2.imread(path + line[2].split('/')[-1])

    # append only valid images; filter out NoneType data
    if img_center is not None:
        img_center = cv2.cvtColor(img_center, cv2.COLOR_RGB2BGR) # convert image color from BGR to RGB
        images.append(img_center)
        measurements.append(steering_center)

    if img_left is not None:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)  # convert image color from BGR to RGB
        images.append(img_left)
        measurements.append(steering_left)

    if img_right is not None:
        img_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)  # convert image color from BGR to RGB
        images.append(img_right)
        measurements.append(steering_right)
       
# convert data into numpy arrays since Keras requires so
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()

# preprocess the images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

# add layers to the network
model.add(Convolution2D(24,5,strides=(2,2),activation='relu'))
model.add(Convolution2D(36,5,strides=(2,2),activation='relu'))
model.add(Convolution2D(48,5,strides=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,activation='relu'))
model.add(Convolution2D(64,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# run the network
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

# save the model to an .h5 file
model.save('model.h5')