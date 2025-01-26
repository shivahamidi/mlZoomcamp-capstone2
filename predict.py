####### Heart Disease Prediction #######

# Necessary import
import pickle
from flask import Flask, request, jsonify


# Model file name
model_file = f'rf_model:40_trees_depth_10_min_samples_leaf_1.bin'

# Open model file to read it
with open(model_file, 'rb') as f_in:
    # Load the model
    One_Hot_encoder, rf = pickle.load(f_in)

# Create the app
app = Flask("heart_disease")

# decorator to link the function to our app
@app.route("/predict", methods = ['POST']) # `POST` method as we send some information about patients
def predict():
    # Get the .json data
    patient = request.get_json()
    
    # One-Hot-Encoding
    X = One_Hot_encoder.transform([patient])
    # Make soft prediction
    y_pred = round(rf.predict_proba(X)[0, 1], 3)
    # Make decision
    heart_disease = y_pred >= 0.5
    
    # Prepare response
    result = {
        "heart_disease_probability": float(y_pred),
        "heart_disease": bool(heart_disease)
    }
    # return the result
    return jsonify(result)
    

# Condition to execute code only if run as a script
if __name__ == "__main__":
    # Run application in debug mode and specifying the localhost
    app.run(debug = True, host = '0.0.0.0', port = 9696)


#---