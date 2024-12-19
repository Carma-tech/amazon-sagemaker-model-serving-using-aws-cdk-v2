import boto3
import json
import requests


########################### Batch Testing Script ###########################
# Replace with your actual API Gateway endpoint
api_url = "https://k9vii2yok7.execute-api.us-east-1.amazonaws.com/MLDemo/text"

# List of test sentences
test_sentences = [
    {"sentence": "The new president has called for an emergency conference for international cooperation."},
    {"sentence": "Baseball is one of the most popular sports in the United States."},
    {"sentence": "Stock investing has higher returns in the long run."},
    {"sentence": "The development of science accelerated the development of mankind."}
]

# Loop through test sentences and send requests
for i, payload in enumerate(test_sentences):
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        print(f"Test {i+1} Response:", response.json())
    else:
        print(f"Test {i+1} Error: {response.status_code}, {response.text}")


###################### This method just for a single input data ###############################
# # Replace with your actual API Gateway endpoint
# api_url = "https://lizwynp710.execute-api.us-east-1.amazonaws.com/MLDemo/text"

# # Request payload
# payload = {
#     "sentence": "The new president has called for an emergency conference for international cooperation."
# }

# # Send POST request
# response = requests.post(api_url, json=payload)

# # Print response
# if response.status_code == 200:
#     print("Response:", response.json())
# else:
#     print(f"Error: {response.status_code}, {response.text}")
