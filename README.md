# Cloud-Native ML Image Classification on AWS

CS346 Project 4 — an end-to-end image classification API built on AWS. A pretrained MobileNetV2 model is deployed to a SageMaker real-time endpoint; AWS Lambda preprocesses incoming images and invokes the endpoint; API Gateway exposes a public HTTP route.

## Architecture

```
   Client
     │  POST /classify  { "image_url": "..." }
     ▼
   API Gateway  ── HTTP API ──► AWS Lambda
                                  │  - download image from URL
                                  │  - resize to 224x224, normalize
                                  │  - reshape to [1, 3, 224, 224]
                                  ▼
                                SageMaker Real-Time Endpoint
                                  │  MobileNetV2 inference (CPU, ml.t2.medium)
                                  ▼
                                JSON response
                                  { "prediction": "...",
                                    "confidence": 0.xx,
                                    "class_id": N }
                                  │
                                  ▼
                                Lambda → API Gateway → Client

   Amazon S3
     └─ project4-ml-model-<account-id>/model/model.tar.gz
           (model.pth + code/inference.py + code/imagenet_classes.json)
```

- **Model**: MobileNetV2, ImageNet pretrained weights (~14 MB)
- **Region**: us-east-1
- **Instance type**: ml.t2.medium (1 instance)
- **Lambda runtime**: Python 3.11, 1024 MB memory, 60s timeout

## Repository Layout

```
.
├── download_model.py              # Task 1 — saves MobileNetV2 weights to model.pth
├── deploy_sagemaker.py            # Task 2 — deploys the SageMaker endpoint
├── test_sagemaker_endpoint.py     # Task 2 — invokes the endpoint directly
├── requirements.txt               # local dev dependencies
├── sagemaker_model/
│   └── code/
│       ├── inference.py           # SageMaker model_fn / input_fn / predict_fn / output_fn
│       └── imagenet_classes.json  # ImageNet label map (1000 classes)
├── lambda_function/
│   ├── lambda_function.py         # Task 3 — Lambda handler that calls SageMaker
│   └── requirements.txt           # pillow, numpy
├── iam/
│   ├── sagemaker-trust-policy.json
│   ├── lambda-trust-policy.json
│   └── lambda-sagemaker-policy.json
├── tests/
│   ├── lambda-test-event.json     # sample API Gateway event for Lambda invoke
│   ├── lambda-output.json         # captured Lambda response
│   ├── latency_test.py            # Task 4 — runs 10 requests against the API
│   └── latency_results.txt        # captured performance results
└── scripts/
    └── deploy.sh                  # consolidated AWS CLI commands for Tasks 1–4
```

## API

**Endpoint**: `POST /classify`
**Content-Type**: `application/json`

**Request body** (either field works):
```json
{ "image_url": "https://example.com/cat.jpg" }
```
or
```json
{ "image_base64": "<base64-encoded image bytes>" }
```

**Response**:
```json
{
  "prediction": "Egyptian cat",
  "confidence": 0.0629730299115181,
  "class_id": 285
}
```

**Example curl**:
```bash
curl -X POST "https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/classify" \
    -H "Content-Type: application/json" \
    -d '{"image_url":"https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"}'
```

## Reproducing the Deployment

`scripts/deploy.sh` contains every AWS CLI command grouped by task. It is a reference — run sections individually rather than executing the whole file.

High-level flow:

1. **Task 1**: download MobileNetV2 weights, package `model.tar.gz`, upload to a new S3 bucket.
2. **Task 2**: create the SageMaker execution role, attach `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`, run `deploy_sagemaker.py`, verify with `aws sagemaker describe-endpoint` (status should be `InService`).
3. **Task 3**: build the Lambda zip in a Linux container (Pillow + NumPy need Linux wheels), create the Lambda execution role with `sagemaker:InvokeEndpoint` permission, deploy the function, wire up API Gateway HTTP API with a POST `/classify` route.
4. **Task 4**: paste the API URL into `tests/latency_test.py` and run it.

## Performance Results

Measured against the deployed `/classify` endpoint with 10 distinct image URLs (end-to-end latency, including API Gateway routing, Lambda execution, image download, preprocessing, and SageMaker inference).

| Metric | Value |
|---|---|
| Average latency | 1.345 s |
| Median latency | 1.242 s |
| Min latency | 1.075 s |
| Max latency | 2.001 s |
| Success rate | 10/10 (HTTP 200) |

Full per-request results in `tests/latency_results.txt`.

### Sample Predictions

| Image | Prediction | Confidence | Latency |
|---|---|---:|---:|
| Cat | Egyptian cat | 0.0630 | 1.379 s |
| Dog 1 | golden retriever | 0.3409 | 1.534 s |
| Dog 2 | Border collie | 0.2407 | 1.077 s |
| Car | grille | 0.1685 | 1.180 s |
| Coffee | espresso | 0.0867 | 2.001 s |
| Apple | pomegranate | 0.2625 | 1.075 s |
| Bird | jacamar | 0.0714 | 1.241 s |
| Laptop | notebook | 0.1012 | 1.242 s |
| Backpack | backpack | 0.1773 | 1.479 s |
| Bicycle | bicycle-built-for-two | 0.1845 | 1.239 s |

## Design Notes

**Why MobileNetV2.** Lightweight enough to run efficiently on CPU (ml.t2.medium), which keeps endpoint cost low. The trade-off is lower top-1 confidence on some inputs compared to larger backbones like ResNet50.

**Why a Linux container for the Lambda zip.** Pillow and NumPy ship native wheels; building the package on macOS produces binaries Lambda's Linux runtime can't load. Using `public.ecr.aws/sam/build-python3.11` produces a compatible package.

**Scaling bottleneck.** API Gateway and Lambda scale horizontally on demand. The single-instance SageMaker endpoint is the first bottleneck under load — production would enable endpoint autoscaling or use a larger instance type.

**Main cost driver.** The SageMaker real-time endpoint is billed continuously while running, regardless of traffic. S3, Lambda, and API Gateway costs are negligible for the testing workload here. The endpoint should be deleted when not in use:

```bash
aws sagemaker delete-endpoint --endpoint-name project4-mobilenetv2-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name project4-mobilenetv2-endpoint
aws sagemaker delete-model --model-name <model-name-from-deploy-output>
```

## Cleanup

To avoid ongoing charges after testing:

```bash
aws sagemaker delete-endpoint --endpoint-name project4-mobilenetv2-endpoint
aws apigatewayv2 delete-api --api-id <api-id>
aws lambda delete-function --function-name project4-classify-image
aws s3 rm s3://project4-ml-model-<account-id> --recursive
aws s3 rb s3://project4-ml-model-<account-id>
aws iam detach-role-policy --role-name Project4SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam detach-role-policy --role-name Project4SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role --role-name Project4SageMakerExecutionRole
aws iam detach-role-policy --role-name Project4LambdaExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role-policy --role-name Project4LambdaExecutionRole --policy-name Project4InvokeSageMakerPolicy
aws iam delete-role --role-name Project4LambdaExecutionRole
```
