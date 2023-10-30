# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:06:09 2021

@author: Reza
"""
import numpy as np 
import keras 

### Generate random training and testing sets 
x_train = np.random.random((1000,20))
y_train = np.random.randint(2,size=(x_train.shape[0],1))
y_train = keras.utils.to_categorical(y_train, num_classes=2)

x_test = np.random.random((100,20))
y_test = np.random.randint(2,size=(x_test.shape[0],1))
y_test = keras.utils.to_categorical(y_test, num_classes=2)

### Defining layers of the MLP model 
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=20))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(2))
model.add(keras.layers.Activation('softmax'))

model.summary()


### Configuring the settings of the training procedure 
model.compile(keras.optimizers.SGD(learning_rate=0.01),
              loss= 'binary_crossentropy',
              metrics= ['accuracy']) 

### Training Model 
model.fit(x_train, y_train, epochs=50, batch_size=64)


### Testing Model 
Output = model.predict(x_test, batch_size=128)
#Pre = model.predict_classes(x_test, batch_size=128)
Pre =np.argmax(model.predict(x_test,batch_size= 128), axis=-1)

Score = model.evaluate(x_test, y_test, batch_size= 128)
print("Test Loss: ", Score[0])
print("Test accuracy: ", Score[1])

              



