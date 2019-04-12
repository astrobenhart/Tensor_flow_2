from __future__ import absolute_import, division, print_function, unicode_literals

#Import Tensorflow and chek that you are using version 2.0.0-alpha0
import tensorflow as tf
print(tf.__version__)

#Download the MNIST dataset
mnist = tf.keras.datasets.mnist

#Split the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Convert the samples from integers to floating-point numbers
x_train, x_test = x_train / 255.0, x_test / 255.0

#Build the tf.keras.Sequential model by stacking layers
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

#Choose an optimizer and loss function used for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train and evaluate model
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

