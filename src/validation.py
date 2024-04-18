def validate_predict(data):
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
