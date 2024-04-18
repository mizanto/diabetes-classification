import os
from flask import Flask, request, jsonify, abort

from model_trainer import load_model, create_model, predict
from validation import validate_predict


MODEL_FILE_PATH = '../models/diabetes_model.pkl'

app = Flask(__name__)

model = None
if os.path.isfile(MODEL_FILE_PATH):
    print('Loading model from file...')
    model = load_model(MODEL_FILE_PATH)
    print('Model loaded.')
else:
    print('Creating model...')
    model = create_model(MODEL_FILE_PATH)
    print('Model created.')


@app.route('/predict', methods=['POST'])
def predict_diabetes():
    """
    Endpoint for predicting diabetes based on input features.

    Request Parameters:
        - age (int): The age of the patient.
        - gender (str): The gender of the patient.
        - polyuria (bool): Whether the patient has polyuria.
        - polydipsia (bool): Whether the patient has polydipsia.
        - sudden_weight_loss (bool): Whether the patient has sudden weight
            loss.
        - weakness (bool): Whether the patient has weakness.
        - polyphagia (bool): Whether the patient has polyphagia.
        - visual_blurring (bool): Whether the patient has visual blurring.
        - irritability (bool): Whether the patient has irritability.
        - partial_paresis (bool): Whether the patient has partial paresis.
        - muscle_stiffness (bool): Whether the patient has muscle stiffness.
        - alopecia (bool): Whether the patient has alopecia.
    """
    try:
        validate_predict(request.json)
    except ValueError as e:
        abort(400, str(e))

    prediction = predict(model, request.json)
    print(f'Prediction: {prediction}')

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(port=5000, debug=True)


# Test JSON
test_json = {
    "age": 35,
    "gender": "male",
    "polyuria": True,
    "polydipsia": False,
    "sudden_weight_loss": True,
    "weakness": False,
    "polyphagia": True,
    "visual_blurring": False,
    "irritability": True,
    "partial_paresis": False,
    "muscle_stiffness": True,
    "alopecia": False
}
