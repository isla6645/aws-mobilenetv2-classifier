import json
import base64
import boto3
import urllib.request
from io import BytesIO
from PIL import Image
import numpy as np

# This is the SageMaker endpoint that was already deployed and tested.
ENDPOINT_NAME = "project4-mobilenetv2-endpoint"
REGION = "us-east-1"

# SageMaker Runtime client is used by Lambda to call the deployed endpoint.
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION)


def preprocess_image(image_bytes):
    """
    Converts raw image bytes into the tensor format expected by MobileNetV2.
    MobileNetV2 expects a 224x224 RGB image normalized with ImageNet statistics.
    Final output shape: [1, 3, 224, 224]
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    arr = np.array(image).astype(np.float32) / 255.0

    # ImageNet normalization values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    # Convert from height-width-channel to channel-height-width
    arr = np.transpose(arr, (2, 0, 1))

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)

    return arr.tolist()


def get_image_bytes(body):
    """
    Accepts either an image URL or a base64-encoded image.

    Example URL request:
        {
            "image_url": "https://example.com/cat.jpg"
        }

    Example base64 request:
        {
            "image_base64": "..."
        }
    """
    if "image_url" in body:
        with urllib.request.urlopen(body["image_url"], timeout=10) as response:
            return response.read()

    if "image_base64" in body:
        return base64.b64decode(body["image_base64"])

    raise ValueError("Request must include either image_url or image_base64.")


def lambda_handler(event, context):
    """
    Main Lambda handler.
    API Gateway sends the HTTP request here.
    Lambda preprocesses the image, invokes SageMaker, and returns JSON.
    """
    try:
        raw_body = event.get("body", "{}")

        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode("utf-8")

        body = json.loads(raw_body)

        image_bytes = get_image_bytes(body)
        tensor = preprocess_image(image_bytes)

        sagemaker_payload = {
            "instances": tensor
        }

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(sagemaker_payload)
        )

        prediction = json.loads(response["Body"].read().decode("utf-8"))

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "class_id": prediction["class_id"]
            })
        }

    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": str(e)
            })
        }
