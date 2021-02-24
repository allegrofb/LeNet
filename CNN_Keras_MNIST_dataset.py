import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) =  datasets.mnist.load_data()

print('x_train:\t{}' .format(x_train.shape))
print('y_train:\t{}' .format(y_train.shape))
print('x_test:\t\t{}'.format(x_test.shape))
print('y_test:\t\t{}'.format(y_test.shape))

# Add a new axis
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train[0].shape, 'image shape')

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# # Convert class vectors to binary class matrices.
# num_classes = 10
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

# Data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Hyperparameters
training_epochs = 5 # Total number of training epochs
learning_rate = 0.001 # The learning rate

# 64 images in a batch
batch_size = 64

def get_model(input_shape,nb_classes):
    # Create a simple model.
    inputs = Input(shape=input_shape)

    # model = Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same")(inputs)
    # model = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(model)
    # model = Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(model)
    # model = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(model)

    # model = Conv2D(20, kernel_size=(5, 5), activation='relu', use_bias=False, input_shape=input_shape)(inputs)
    model = Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=input_shape)(inputs)
    model = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(model)
    # model = Conv2D(50, kernel_size=(5, 5), strides=(1, 1), use_bias=False, activation='relu')(model)
    model = Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='relu')(model)
    model = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(model)
    model = Flatten()(model)
    model = Dense(512, activation='relu')(model)
    model = Dense(nb_classes, activation='softmax')(model)

    model = Model(inputs, model)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                loss=categorical_crossentropy,
                metrics=['accuracy'])
    # model.compile(optimizer='adam',
    #             loss=categorical_crossentropy,
    #             metrics=['accuracy'])

    return model

model = get_model(x_train[0].shape, num_classes)

model.summary()

results = model.fit(
 x_train, y_train,
 epochs= training_epochs,
 batch_size = batch_size,
 validation_data = (x_test, y_test),
 verbose = 2
)

model.save("lenet-mnist.h5",save_format='h5')
model.save("lenet-mnist")

predictions = model.predict(x_test)
prediction_values = np.argmax(predictions,axis=1)

# set up the figure
fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(10):
    ax = fig.add_subplot(6, 20, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[i,:].reshape((28,28)),cmap=plt.cm.gray_r, interpolation='nearest')
    
    print(predictions[i])
    print(y_test[i])

    # label the image with the target value
    ax.text(0, 7, str(prediction_values[i]))

plt.show()



# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 24, 24, 20)        520
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 12, 12, 20)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 8, 8, 50)          25050
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 4, 4, 50)          0
# _________________________________________________________________
# flatten (Flatten)            (None, 800)               0
# _________________________________________________________________
# dense (Dense)                (None, 512)               410112
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                5130
# =================================================================
# Total params: 440,812
# Trainable params: 440,812
# Non-trainable params: 0
# _________________________________________________________________

# [3.2368231e-12 5.9610055e-09 2.6025751e-10 1.7926251e-07 7.6221340e-09
#  1.1218865e-10 4.6206171e-16 9.9999976e-01 4.1890522e-10 2.4602532e-08]
# [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
# [2.07651922e-08 5.11855148e-07 9.99999523e-01 6.64670792e-13
#  4.59540088e-12 4.76961606e-17 1.30706695e-11 2.12121085e-11
#  1.79697804e-12 9.56222731e-13]
# [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# [2.0561542e-08 9.9998951e-01 2.9316614e-09 1.1308314e-10 5.3360109e-06
#  3.0656246e-08 1.6623510e-08 3.7285338e-06 1.4392884e-06 5.4820216e-08]
# [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# [9.9998975e-01 2.3903615e-10 1.7295481e-08 8.6633162e-10 9.0521551e-10
#  2.5081048e-09 1.0082917e-05 9.4750812e-09 3.7891976e-08 7.4535812e-08]
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [2.7699254e-10 1.0042052e-07 1.7405121e-09 5.6755450e-10 9.9996185e-01
#  1.3419809e-09 3.8595434e-09 8.9507374e-08 3.4388652e-08 3.7887185e-05]
# [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# [9.5685468e-09 9.9999440e-01 1.6876286e-09 5.0253066e-12 1.0242553e-06
#  3.0320427e-10 1.3462084e-09 4.4252993e-06 8.4877669e-08 5.7804503e-09]
# [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# [1.5078443e-15 3.7184302e-07 4.4040674e-12 5.2925345e-13 9.9997926e-01
#  4.6275744e-10 2.6883554e-12 1.9934250e-05 1.9409286e-07 2.7468863e-07]
# [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# [2.4060391e-12 1.4185301e-09 1.4710755e-10 1.4648073e-09 6.4656570e-06
#  3.4081085e-11 2.2963751e-16 3.5818321e-11 2.0453958e-06 9.9999154e-01]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
# [5.9776418e-08 1.1679960e-09 4.8007268e-11 4.3623433e-10 2.5408733e-08
#  9.5985955e-01 3.6536176e-02 2.7357709e-09 3.6040759e-03 2.1010210e-08]
# [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# [2.4354451e-12 3.0961941e-11 3.0669692e-12 4.1202039e-10 3.4352874e-05
#  5.6537223e-11 9.2835827e-15 5.9310435e-07 5.0358244e-07 9.9996459e-01]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
























