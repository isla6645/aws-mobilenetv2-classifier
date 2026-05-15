import json
import boto3
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import time

ENDPOINT_NAME = "project4-mobilenetv2-endpoint"
REGION = "us-east-1"

IMAGE_URL = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"

def preprocess_image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize((224, 224))

    arr = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr.tolist()

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

payload = {
    "instances": preprocess_image_from_url(IMAGE_URL)
}

start = time.time()

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload)
)

end = time.time()

result = json.loads(response["Body"].read().decode("utf-8"))

print(json.dumps(result, indent=2))
print(f"Latency seconds: {end - start:.3f}")
