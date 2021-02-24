import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

x_train = x_train[0:100]
y_train = y_train[0:100]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train[0].shape, 'image shape')


# Add a new axis
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train[0].shape, 'image shape')


# Convert class vectors to binary class matrices.

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# Data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# LeNet-5 model
class LeNet(Sequential):
  def __init__(self, input_shape, nb_classes):
    super().__init__()

    # self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
    self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
    self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    self.add(Flatten())
    # self.add(Dense(120, activation='tanh'))
    # self.add(Dense(84, activation='tanh'))
    self.add(Dense(120, activation='relu'))
    self.add(Dense(84, activation='relu'))
    self.add(Dense(nb_classes, activation='softmax'))

    self.compile(optimizer='adam',
                loss=categorical_crossentropy,
                metrics=['accuracy'])

model = LeNet(x_train[0].shape, num_classes)

model.summary()

# # Place the logs in a timestamped subdirectory
# # This allows to easy select different training runs
# # In order not to overwrite some data, it is useful to have a name with a timestamp
# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# # Specify the callback object
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# tf.keras.callback.TensorBoard ensures that logs are created and stored
# We need to pass callback object to the fit method
# The way to do this is by passing the list of callback objects, which is in our case just one


model.fit(x_train, y=y_train, 
          epochs=20, 
          validation_data=(x_test, y_test), 
        #   callbacks=[tensorboard_callback],
          verbose=0)


model.save("lenet.h5",save_format='h5')
model.save("lenet")

# %tensorboard --logdir logs/fit


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

prediction_values = model.predict_classes(x_test)

# set up the figure
fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the images: each image is 28x28 pixels
for i in range(50):
  ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
  ax.imshow(x_test[i,:].reshape((28,28)),cmap=plt.cm.gray_r, interpolation='nearest')
  
  if prediction_values[i] == np.argmax(y_test[i]):
    # label the image with the blue text
    ax.text(0, 7, class_names[prediction_values[i]], color='blue')
  else:
    # label the image with the red text
    ax.text(0, 7, class_names[prediction_values[i]], color='red')

plt.show()





































