### Heart Disease Prediction App test ###

# import necessary library
import requests

# url address for making predictions
url = "http://localhost:9696/predict"
# New patient information
patient = {'age': 17,
           'sex': 'male',
           'cp': 'asymptomatic',
           'trestbps': 67.0,
           'chol': 133.0,
           'fbs': 'low_fbs',
           'restecg': 'normal',
           'thalachh': 82.0,
           'exang': 'yes',
           'oldpeak': 1.7,
           'slope': 'upsloping',
           'ca': 'three_vessels',
           'thal': 'normal'}

# send a request for making predictions
response = requests.post(url, json = patient).json()
# Print the response
print(response)
# Define a treatment if necessary
if response['heart_disease']:
    print('Define a treatment for the patient-test.')
else:
    print('The patient seems healthy: no treatment needed.')
    
# ---