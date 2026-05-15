import boto3
from sagemaker.pytorch import PyTorchModel

REGION = "us-east-1"
ENDPOINT_NAME = "project4-mobilenetv2-endpoint"

account_id = boto3.client("sts").get_caller_identity()["Account"]
bucket_name = f"project4-ml-model-{account_id}"

role_arn = boto3.client("iam").get_role(
    RoleName="Project4SageMakerExecutionRole"
)["Role"]["Arn"]

model_data = f"s3://{bucket_name}/model/model.tar.gz"

print("Model artifact:", model_data)
print("Role ARN:", role_arn)
print("Endpoint name:", ENDPOINT_NAME)

pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role_arn,
    framework_version="2.0.1",
    py_version="py310",
    entry_point="inference.py",
    source_dir="sagemaker_model/code"
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=ENDPOINT_NAME
)

print("Endpoint deployed successfully.")
