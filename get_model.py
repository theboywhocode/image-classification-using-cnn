#Let's build a model to train our machine 

import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#Code to save json file in your system after training, this will help you in deploying this model on web
def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')
    return

#Code for the deep neural network is written here
def get_model(num_classes=10):
    weight_decay = 1e-4
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())



    model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(256))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(num_classes, activation='softmax'))

    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

    return model

  if __name__ == '__main__':
      save_model(get_model())
