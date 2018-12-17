import tensorflow as tf
from LastData import *
from tensorflow.keras.utils import to_categorical

#data, labels = loadFullSignals()
X_train, X_test, y_train, y_test = skms.train_test_split(data, labels, test_size=0.2)

p=1
X_train, X_test, y_train, y_test = (X_train[:int(p*len(y_train))], X_test, y_train[:int(p*len(y_train))], y_test)

signalsize = 178 # evt 4097 om loadFullSignals vert køyrd
features = 1

X_train=X_train.reshape(-1,signalsize,features)
X_test=X_test.reshape(-1,signalsize,features)

model = tf.keras.Sequential()
y_train = y_train.reshape(-1, features)
y_test = y_test.reshape(-1, features)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.add(tf.keras.layers.LSTM(128, input_shape=(signalsize,features), return_sequences=True))

model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64))

model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['binary_accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=[X_test, y_test])
print('accuracy: ', model.evaluate(X_test, y_test))

model.save('network.h5')
