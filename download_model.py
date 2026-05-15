import torch
from torchvision import models

# Load MobileNetV2 with pretrained ImageNet weights.
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

# Save only the model weights.
torch.save(model.state_dict(), "sagemaker_model/model.pth")

print("Saved MobileNetV2 model to sagemaker_model/model.pth")
