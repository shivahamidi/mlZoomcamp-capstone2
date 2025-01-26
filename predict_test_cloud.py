### Heart Disease Prediction App test ###

# import necessary library
import requests

# host address provided by Elastic BeansTalk 
host = 'heart-prediction-app-env.eba-zpm2tfpu.us-east-1.elasticbeanstalk.com'
# url address for making predictions
url = f"http://{host}/predict"
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
