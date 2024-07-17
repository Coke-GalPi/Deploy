from flask import Flask, request, jsonify, render_template
import onnxruntime as rt
import numpy as np

app = Flask(__name__)

# Cargar los modelos ONNX
sessRF = rt.InferenceSession("data/random_forest_model.onnx")
sessDT = rt.InferenceSession("data/decision_tree_model.onnx")
sessFNN = rt.InferenceSession("data/feedforward_nn_model.onnx")
sessFNN2 = rt.InferenceSession("data/FNNwoutES.onnx")

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# Pagina de Random Forest
@app.route('/randomForest')
def randomForest():
    return render_template('randomForest.html')
@app.route('/predictRF', methods=['POST'])
def predictRF():
    data = request.get_json(force=True)
    features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
    input_name = sessRF.get_inputs()[0].name
    prediction = sessRF.run(None, {input_name: features})[0]
    if prediction.item() == 0:
        prediction = 'On Time'
    elif prediction.item() == 1:
        prediction = 'Delayed'
    return jsonify({'prediction': prediction})

# Pagina de Decision Tree
@app.route('/decisionTree')
def decisionTree():
    return render_template('decisionTree.html')
@app.route('/predictDT', methods=['POST'])
def predictDT():
    data = request.get_json(force=True)
    features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
    input_name = sessDT.get_inputs()[0].name
    prediction = sessDT.run(None, {input_name: features})[0]
    if prediction.item() == 0:
        prediction = 'On Time'
    elif prediction.item() == 1:
        prediction = 'Delayed'
    return jsonify({'prediction': prediction})

# Pagina de Feedforward Neural Network 1
@app.route('/feedforwardNN')
def feedforwardNN():
    return render_template('feedforwardNN.html')
@app.route('/predictFNN', methods=['POST'])
def predictFNN():
    data = request.get_json(force=True)
    features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
    input_name = sessFNN.get_inputs()[0].name
    prediction = sessFNN.run(None, {input_name: features})[0]
    prediction = int(prediction.item())
    return jsonify({'prediction': prediction})

# Pagina de Feedforward Neural Network 2
@app.route('/feedforwardNeuralN')
def feedforwardNeuralN():
    return render_template('feedforwardNeuralN.html')
@app.route('/predictFNN_2', methods=['POST'])
def predictFNN_2():
    data = request.get_json(force=True)
    features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
    input_name = sessFNN2.get_inputs()[0].name
    prediction = sessFNN2.run(None, {input_name: features})[0]
    prediction = int(prediction.item())
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)