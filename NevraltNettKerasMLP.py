import tensorflow as tf
from LastData import *
from tensorflow.keras.utils import to_categorical

(X_train, X_test, y_train, y_test), data = featureExtraction(data, False)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(247, activation='softmax', input_dim=X_train.shape[1]))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(359, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.04, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # adam er best
print(model.summary())
model.fit(X_train, y_train, epochs=191, batch_size=32, validation_data=[X_test, y_test])

print('accuracy: ', model.evaluate(X_test, y_test))

model.save('network.h5')
