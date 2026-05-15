#!/usr/bin/env bash
# End-to-end deployment commands for Project 4.
# This is a reference script — run sections individually, not all at once.
# Fill in your own region/bucket details as needed.

set -e

REGION="us-east-1"
ENDPOINT_NAME="project4-mobilenetv2-endpoint"
LAMBDA_NAME="project4-classify-image"
API_NAME="project4-image-classifier-api"

# -----------------------------------------------------------------------------
# Task 1: Model preparation and S3 upload
# -----------------------------------------------------------------------------

# Download MobileNetV2 weights locally
python download_model.py

# Generate ImageNet class labels file
mkdir -p sagemaker_model/code
python - <<'EOF'
import json
from torchvision.models import MobileNet_V2_Weights
categories = MobileNet_V2_Weights.DEFAULT.meta["categories"]
labels = {str(i): name for i, name in enumerate(categories)}
with open("sagemaker_model/code/imagenet_classes.json", "w") as f:
    json.dump(labels, f)
EOF

# Package the model artifact
cd sagemaker_model
tar -czf ../model.tar.gz .
cd ..
tar -tzf model.tar.gz | head

# Create S3 bucket and upload
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="project4-ml-model-$ACCOUNT_ID"

aws s3 mb "s3://$BUCKET_NAME" --region "$REGION"
aws s3 cp model.tar.gz "s3://$BUCKET_NAME/model/model.tar.gz"
aws s3 ls "s3://$BUCKET_NAME/model/"

# -----------------------------------------------------------------------------
# Task 2: Deploy SageMaker endpoint
# -----------------------------------------------------------------------------

# Create SageMaker execution role
aws iam create-role \
    --role-name Project4SageMakerExecutionRole \
    --assume-role-policy-document file://iam/sagemaker-trust-policy.json

aws iam attach-role-policy \
    --role-name Project4SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name Project4SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Deploy the endpoint
python deploy_sagemaker.py

# Verify status
aws sagemaker describe-endpoint \
    --endpoint-name "$ENDPOINT_NAME" \
    --query "EndpointStatus"

# Test the endpoint directly
python test_sagemaker_endpoint.py

# -----------------------------------------------------------------------------
# Task 3: Lambda + API Gateway
# -----------------------------------------------------------------------------

# Build the Lambda deployment package using a Linux-compatible container
cd lambda_function
rm -rf package lambda_function.zip
mkdir package
docker run --rm \
    -v "$PWD":/var/task \
    public.ecr.aws/sam/build-python3.11 \
    /bin/sh -c "pip install -r requirements.txt -t package"
cp lambda_function.py package/
cd package
zip -r ../lambda_function.zip .
cd ../..

# Create Lambda execution role
aws iam create-role \
    --role-name Project4LambdaExecutionRole \
    --assume-role-policy-document file://iam/lambda-trust-policy.json

aws iam attach-role-policy \
    --role-name Project4LambdaExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam put-role-policy \
    --role-name Project4LambdaExecutionRole \
    --policy-name Project4InvokeSageMakerPolicy \
    --policy-document file://iam/lambda-sagemaker-policy.json

LAMBDA_ROLE_ARN=$(aws iam get-role \
    --role-name Project4LambdaExecutionRole \
    --query "Role.Arn" --output text)

# Create the Lambda function
aws lambda create-function \
    --function-name "$LAMBDA_NAME" \
    --runtime python3.11 \
    --role "$LAMBDA_ROLE_ARN" \
    --handler lambda_function.lambda_handler \
    --timeout 60 \
    --memory-size 1024 \
    --zip-file fileb://lambda_function/lambda_function.zip

# Test Lambda directly
aws lambda wait function-updated --function-name "$LAMBDA_NAME"
aws lambda invoke \
    --function-name "$LAMBDA_NAME" \
    --cli-binary-format raw-in-base64-out \
    --payload file://tests/lambda-test-event.json \
    tests/lambda-output.json
cat tests/lambda-output.json | jq

# Create HTTP API
API_ID=$(aws apigatewayv2 create-api \
    --name "$API_NAME" \
    --protocol-type HTTP \
    --query "ApiId" --output text)

LAMBDA_ARN=$(aws lambda get-function \
    --function-name "$LAMBDA_NAME" \
    --query "Configuration.FunctionArn" --output text)

INTEGRATION_ID=$(aws apigatewayv2 create-integration \
    --api-id "$API_ID" \
    --integration-type AWS_PROXY \
    --integration-uri "$LAMBDA_ARN" \
    --payload-format-version 2.0 \
    --query "IntegrationId" --output text)

aws apigatewayv2 create-route \
    --api-id "$API_ID" \
    --route-key "POST /classify" \
    --target "integrations/$INTEGRATION_ID"

aws apigatewayv2 create-stage \
    --api-id "$API_ID" \
    --stage-name prod \
    --auto-deploy

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws lambda add-permission \
    --function-name "$LAMBDA_NAME" \
    --statement-id apigateway-project4-permission \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:$REGION:$ACCOUNT_ID:$API_ID/*/*/classify"

API_URL="https://$API_ID.execute-api.$REGION.amazonaws.com/prod/classify"
echo "API endpoint: $API_URL"

# Test the public API
curl -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d '{"image_url":"https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"}'

# -----------------------------------------------------------------------------
# Task 4: Performance testing
# -----------------------------------------------------------------------------

# Edit tests/latency_test.py and paste the API_URL value before running
python tests/latency_test.py > tests/latency_results.txt
cat tests/latency_results.txt
