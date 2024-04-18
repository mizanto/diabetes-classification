import os
from flask import Flask, request, jsonify, abort

from model_trainer import load_model, create_model, predict


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
        _validate_predict(request.json)
    except ValueError as e:
        abort(400, str(e))

    prediction = predict(model, request.json)
    print(f'Prediction: {prediction}')

    return jsonify({'prediction': prediction})


def _validate_predict(data):
    fields = [
        'age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
        'weakness', 'polyphagia', 'visual_blurring', 'irritability',
        'partial_paresis', 'muscle_stiffness', 'alopecia'
    ]

    boolean_fields = [
        'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness',
        'polyphagia', 'visual_blurring', 'irritability', 'partial_paresis',
        'muscle_stiffness', 'alopecia'
    ]

    missing_fields = [field for field in fields if field not in data]
    if missing_fields:
        raise ValueError(f'Missing fields in the request: {missing_fields}')

    if not isinstance(data['age'], int) or data['age'] <= 0:
        raise ValueError('Invalid age. Age must be a positive integer.')

    if data['gender'] not in ['Male', 'Female']:
        raise ValueError('Invalid gender.')

    for field in boolean_fields:
        if not isinstance(data[field], bool):
            raise ValueError(f'Invalid value for {field}. Must be a boolean.')


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
