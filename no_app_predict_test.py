#!/usr/bin/env python
# coding: utf-8

#### Necessary import
import pickle # to manipulate models

### Load the model
# Name of the model
input_file = 'rf_model:40_trees_depth_10_min_samples_leaf_1.bin'

# Open file to read it
with open(input_file, 'rb') as f_in:
    # Load the model
    One_Hot_encoder, rf = pickle.load(f_in)

### Test the model
# Random patient information
patient = {'age': 37,
           'sex': 'male',
           'cp': 'typical_angina',
           'trestbps': 137.0,
           'chol': 193.0,
           'fbs': 'low_fbs',
           'restecg': 'st_t_wave_abnormality',
           'thalachh': 112.0,
           'exang': 'no',
           'oldpeak': 3.7,
           'slope': 'downsloping',
           'ca': 'no_vessel',
           'thal': 'fixed_defect'}
# One-Hot-Encoding
X_i = One_Hot_encoder.transform([patient])
# Make predictions
y_i_pred = round(rf.predict_proba(X_i)[0, 1], 3)

# Print customer info and the model's prediction
print('input data:', patient)
print('output information:', y_i_pred)
# Define a treatment if necessary
if y_i_pred:
    print('Define a treatment for the patient-test.')
    
# ---