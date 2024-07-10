from flask import Flask, request, jsonify, render_template
import onnxruntime as rt
import numpy as np

app = Flask(__name__)

# Cargar los modelos ONNX
sessRF = rt.InferenceSession("data/random_forest_model.onnx")
sessDT = rt.InferenceSession("data/decision_tree_model.onnx")

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/randomForest')
def randomForest():
    return render_template('randomForest.html')

@app.route('/decisionTree')
def decisionTree():
    return render_template('decisionTree.html')

@app.route('/predictRF', methods=['POST'])
def predictRF():
    data = request.get_json(force=True)
    features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
    input_name = sessRF.get_inputs()[0].name
    prediction = sessRF.run(None, {input_name: features})[0]
    return jsonify({'prediction': prediction.tolist()})

@app.route('/predictDT', methods=['POST'])
def predictDT():
    data = request.get_json(force=True)
    features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
    input_name = sessDT.get_inputs()[0].name
    prediction = sessDT.run(None, {input_name: features})[0]
    print(f'prediction: {prediction}')
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)