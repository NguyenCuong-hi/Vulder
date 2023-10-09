
import numpy as np
import tensorflow as tf
from keras import Input, Model, Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pickle as pk

# data = pickle.load(open("D:/folder_data/neurips_parsed/data.npy", 'rb'))
data = np.load("D:/folder_data/neurips_parsed/data.npy")

labels = np.random.randint(2, size=203511)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)


scale = MinMaxScaler();
x_train_scale = scale.fit_transform(x_train)
x_test_scale = scale.fit_transform(x_test)

time_steps = x_train_scale
input_dim = x_train.shape[1]


model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(x_train, input_dim, 100)))
model.add(Dense(1))
model.build()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
print(x_train_scale.shape)
print(time_steps)
print(input_dim)
model.fit(x_train_scale, y_train, epochs=10, verbose=0)


