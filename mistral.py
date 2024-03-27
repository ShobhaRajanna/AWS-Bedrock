import boto3
import json

prompt_data = """
Things to eat in mangalore for vegetarians
"""
bedrock = boto3.client(service_name ="bedrock-runtime")
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_tokens":512,
    "temperature":0.5,
    "top_p":0.98,
}

body = json.dumps(payload)
model_id = "mistral.mixtral-8x7b-instruct-v0:1"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response=json.loads(response.get('body').read())
for output in response['outputs']:
    print(output['text'])



