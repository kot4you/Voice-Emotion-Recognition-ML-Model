import pickle
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.models import model_from_json


# open the saved test data samples
with open('Saved/x_testcnn.pickle', 'rb') as f:
    x_testcnn = pickle.load(f)

with open('Saved/y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)
# open the saved encoder
with open('Saved/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

# open the model
json_file = open('Saved/CNN_model.json', 'r')

loaded_model_json = json_file.read()
json_file.close()

# load the model
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Saved/best_model.weights.h5")
print("Loaded model from disk")

# compile the loaded model
loaded_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# evaluate the score
score = loaded_model.evaluate(x_testcnn, y_test)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# Evaluate the model
pred_test0 = loaded_model.predict(x_testcnn)
y_pred0 = encoder.inverse_transform(pred_test0)
y_test0 = encoder.inverse_transform(y_test)

print(classification_report(y_test0, y_pred0))
