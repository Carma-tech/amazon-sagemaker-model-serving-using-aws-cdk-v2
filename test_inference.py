import boto3
import json
import requests


########################### Batch Testing Script ###########################
# Replace with your actual API Gateway endpoint
api_url = "https://1y9whl8565.execute-api.us-east-1.amazonaws.com/MLDemo/text"

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

# from sagemaker import image_uris
# container = image_uris.retrieve(framework='blazingtext', region='us-east-1')
# print(container) # 811284229777.dkr.ecr.us-east-1.amazonaws.com/blazingtext:1