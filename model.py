import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding
from keras.layers import LSTM, BatchNormalization, GRU
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import tensorflow.keras.layers as L
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# load data and fill missing values
data = pd.read_csv('Saved/features.csv')
data = data.fillna(0)

# Initialize the data sets
X = data.iloc[:, :-1].values
Y = data['Emotions'].values

# Encode the target feature array
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

# Split the data for training
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, random_state=42, test_size=0.2, shuffle=True)

# Reshape for CNN
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Load the scaler and scale the train feauters set.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Set the checkpoint config and weigth saving.
model_checkpoint = ModelCheckpoint(
    'best_model.weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True)

# Configure early stopping
early_stop = EarlyStopping(monitor='val_accuracy',
                           mode='auto', patience=5, restore_best_weights=True)
# Configure learning rate reduction
lr_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# Expand dimensions for CNN
x_traincnn = np.expand_dims(x_train, axis=2)
x_testcnn = np.expand_dims(x_test, axis=2)

# Make model architecutre 5 convolutional layers with batch normalization, pooling and dropout and a dense softmax layer.
model = tf.keras.Sequential([
    L.Conv1D(512, kernel_size=5, strides=1, padding='same',
             activation='relu', input_shape=(X_train.shape[1], 1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),

    L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),
    Dropout(0.2),

    L.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),

    L.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),
    Dropout(0.2),

    L.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3, strides=2, padding='same'),
    Dropout(0.2),

    L.Flatten(),
    L.Dense(512, activation='relu'),
    L.BatchNormalization(),
    L.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Start the training
history = model.fit(x_traincnn, y_train, epochs=50, validation_data=(
    x_testcnn, y_test), batch_size=64, callbacks=[early_stop, lr_reduction, model_checkpoint])

# Print accuracy after training
print("Accuracy of our model on test data : ",
      model.evaluate(x_testcnn, y_test)[1]*100, "%")

# Visualize the training process
epochs = [i for i in range(50)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20, 6)
ax[0].plot(epochs, train_loss, label='Training Loss')
ax[0].plot(epochs, test_loss, label='Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs, train_acc, label='Training Accuracy')
ax[1].plot(epochs, test_acc, label='Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()


# Predicting on test data.
pred_test0 = model.predict(x_testcnn)
y_pred0 = encoder.inverse_transform(pred_test0)
y_test0 = encoder.inverse_transform(y_test)

# Check on some data sample
df0 = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df0['Predicted Labels'] = y_pred0.flatten()
df0['Actual Labels'] = y_test0.flatten()

print(df0.head(10))

# Make the evaluation. Creat confustion matrix and classification report
cm = confusion_matrix(y_test0, y_pred0)
plt.figure(figsize=(12, 10))
cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[
                  i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues',
            linewidth=1, annot=True, fmt='.2f')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()
print(classification_report(y_test0, y_pred0))

# Save the model in a json file
model_json = model.to_json()
with open("Saved/CNN_model.json", "w") as json_file:
    json_file.write(model_json)

# Save the weigths
model.save_weights("Saved/CNN_model.weights.h5")
print("Saved model to disk")

# Saving the scaler
with open('Saved/scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)

# Saving the encoder
with open('Saved/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

# Saving the split data sets
with open('Saved/x_traincnn.pickle', 'wb') as f:
    pickle.dump(x_traincnn, f)

with open('Saved/y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)

with open('Saved/x_testcnn.pickle', 'wb') as f:
    pickle.dump(x_testcnn, f)

with open('Saved/y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f)
