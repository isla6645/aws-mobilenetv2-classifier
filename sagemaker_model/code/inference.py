import json
import torch
import torch.nn.functional as F
from torchvision import models

# Load ImageNet class labels packaged with the model.
with open("/opt/ml/model/code/imagenet_classes.json", "r") as f:
    IMAGENET_CLASSES = json.load(f)

def model_fn(model_dir):
    """
    Loads the MobileNetV2 model when the SageMaker container starts.
    """
    model = models.mobilenet_v2(weights=None)
    model.load_state_dict(
        torch.load(f"{model_dir}/model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

def input_fn(request_body, content_type):
    """
    Converts JSON input from Lambda into a PyTorch tensor.
    Expected input shape: [1, 3, 224, 224]
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)
    tensor = torch.tensor(data["instances"], dtype=torch.float32)
    return tensor

def predict_fn(input_data, model):
    """
    Runs MobileNetV2 inference and returns the top prediction.
    """
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, class_id = torch.max(probabilities, dim=0)

    class_index = int(class_id.item())

    return {
        "class_id": class_index,
        "prediction": IMAGENET_CLASSES[str(class_index)],
        "confidence": float(confidence.item())
    }

def output_fn(prediction, accept):
    """
    Returns the prediction as JSON.
    """
    return json.dumps(prediction), "application/json"
