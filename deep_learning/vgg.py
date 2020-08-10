import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

ytrain = tf.keras.utils.to_categorical(y_train)
ytest = tf.keras.utils.to_categorical(y_test)

labels = ["airplane", "automovile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("Training images")
for category in [0,1,2,3,4,5,6,7,8,9]:
    index = np.where(y_train == [category])[0][0]
    plt.subplot(2, 5, 0 + 1 + category)
    plt.imshow(x_train[index])
    plt.title(labels[category])
# plt.show()

print("Test images")
for category in [0,1,2,3,4,5,6,7,8,9]:
    index = np.where(y_test == [category])[0][0]
    plt.subplot(2, 5, 0 + 1 + category)
    plt.imshow(x_test[index])
    plt.title(labels[category])
# plt.show()

for category in [0,1,2,3,4,5,6,7,8,9]:
    amount = len(np.where(y_train == [category])[0])
    print("Training images for", labels[category], ":", amount)

print()
for category in [0,1,2,3,4,5,6,7,8,9]:
    amount = len(np.where(y_test == [category])[0])
    print("Training images for", labels[category], ":", amount)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

def get_vgg_model(bn=None, pooling="max", conv_act="relu", fc_act="relu", conv_drop=None, fc_drop=None):
  model = tf.keras.models.Sequential()

  #Bloque1 - feature maps - Convolutional layer:
  model.add(tf.keras.layers.Conv2D(input_shape=(32,32,3),filters=32,kernel_size=(3,3),padding="same", activation=conv_act))
  # model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation=conv_act))
  
  #BN1 - Batch Normalization after
  if bn == 'after':
      model.add(tf.keras.layers.BatchNormalization(axis = -1))
  if conv_drop:
        model.add(tf.keras.layers.Dropout(conv_drop))
  model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation=conv_act))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))) 

  #BN2 - Batch Normalization before
  if bn == 'before':
    model.add(tf.keras.layers.BatchNormalization(axis = -1))
  
  if conv_drop:
        model.add(tf.keras.layers.Dropout(conv_drop))

  #Bloque2 - feature maps - Convolutional layer:
  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=conv_act))
  
  #BN1 - Batch Normalization after
  if bn == 'after':
      model.add(tf.keras.layers.BatchNormalization(axis = -1))

  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=conv_act))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

  #Capas densas - feature maps - Fully Connected layers:
  model.add(tf.keras.layers.Dense(units=512,activation = fc_act))

  #BN3 - Batch Normalization before
  if bn == 'before':
    model.add(tf.keras.layers.BatchNormalization(axis = -1))
 
  if fc_drop:
        model.add(tf.keras.layers.Dropout(fc_drop))

  model.add(tf.keras.layers.Dense(units=10,activation = fc_act))

  #BN3 - Batch Normalization after
  if bn == 'after':
    model.add(tf.keras.layers.BatchNormalization(axis = -1))

  model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
  
  return model

def build_model(bn=None, pooling="max", conv_act="relu", fc_act="relu", conv_drop=None, fc_drop=None):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=conv_act))
    if bn == "after":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))
    if conv_drop:
        model.add(tf.keras.layers.Dropout(conv_drop))

    if bn == "before":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=conv_act))
    if bn == "after":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))

    if pooling == "max":
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    elif pooling == "average":
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

    if bn == "before":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=conv_act))
    if bn == "after":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))
    if conv_drop:
        model.add(tf.keras.layers.Dropout(conv_drop))

    if bn == "before":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=conv_act))
    if bn == "after":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))

    if pooling == "max":
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    elif pooling == "average":
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
    if conv_drop:
        model.add(tf.keras.layers.Dropout(conv_drop))

    model.add(tf.keras.layers.Flatten())
    if bn == "before":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))
    model.add(tf.keras.layers.Dense(units=512, activation=fc_act))
    if bn == "after":
        model.add(tf.keras.layers.BatchNormalization(axis = -1))

    if fc_drop:
        model.add(tf.keras.layers.Dropout(fc_drop))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    return model

configs = [
           [1, 256, 0.01, True], # Base configuration
        #    [3, 16, 0.01, True], # Using smaller batch size
        #    [3, 32, 0.005, True], # Using smaller learning rate
        #    [3, 16, 0.005, True], # Using both smaller batch size and learning rate
        #    [2, 32, 0.01, False] # Deactivating Nerestov
]

for c in configs:
  n_epochs=c[0]
  n_batch=c[1]
  learning_rate=c[2]
  opt = tf.keras.optimizers.SGD(lr=learning_rate, decay=learning_rate / n_epochs, momentum=0.9, nesterov=c[3])
  model = build_model()
  model.compile(optimizer=opt, 
   loss='categorical_crossentropy',
   metrics=['accuracy'])
  
  print("Training model with epochs: {}, batch size: {}, learning_rate: {}, nerestov: {}".format(n_epochs, n_batch, learning_rate, c[3]))
  t = time.process_time()
  H = model.fit(x_train,ytrain, validation_data=(x_test,ytest), 
                epochs=n_epochs, batch_size=n_batch)
  print("Elapsed_time: ", time.process_time() - t)
  
  # Print report
  generate_report(model, H, x_test, ytest, n_batch, n_epochs, labels)
model.summary()
