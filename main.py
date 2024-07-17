from flask import Flask, request, jsonify, render_template
import onnxruntime as rt
import numpy as np
import os

# Crear la aplicacion Flask
app = Flask(__name__)

# Cargar los modelos ONNX
sessRF = rt.InferenceSession("data/random_forest_model.onnx")
sessDT = rt.InferenceSession("data/decision_tree_model.onnx")

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))