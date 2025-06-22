from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import os

app = Flask(__name__)

# Load model and dataset
model_path = 'model/predictor_model.h5'
data_path = 'model/training_data.csv'

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = None

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    data = pd.DataFrame({'crash_point': []})

window_size = 5

def prepare_data():
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data['crash_point'][i:i+window_size])
        y.append(data['crash_point'][i+window_size])
    if len(X) == 0:
        return None, None
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def train_model():
    global model
    X, y = prepare_data()
    if X is not None:
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, verbose=0)
        model.save(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'predict' in request.form:
            try:
                input_sequence = request.form.get('sequence')
                input_list = [float(x.strip()) for x in input_sequence.split(',')]
                input_array = np.array(input_list).reshape((1, window_size, 1))
                prediction = model.predict(input_array)[0][0]
                prediction = round(prediction, 2)
            except:
                prediction = "Invalid input."
        elif 'learn' in request.form:
            try:
                actual_value = float(request.form.get('actual'))
                global data
                new_row = {'crash_point': actual_value}
                data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
                data.to_csv(data_path, index=False)
                train_model()
                prediction = "Model retrained with new data!"
            except:
                prediction = "Error updating data."
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    if model is None:
        train_model()
    app.run(host='0.0.0.0', port=5000)
