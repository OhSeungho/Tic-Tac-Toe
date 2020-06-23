import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense

dataset = np.genfromtxt('data.csv', delimiter=',', dtype=np.float32)

np.random.shuffle(dataset)

x_data = dataset[:, 0:-2]
y_data = dataset[:, 9:]

train_num = 576
test_num = 383

x_train = x_data[: train_num]
y_train = y_data[: train_num]

x_test = x_data[train_num : ]
y_test = y_data[train_num : ]

model = Sequential()
model.add(Dense(100, input_dim=9, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
