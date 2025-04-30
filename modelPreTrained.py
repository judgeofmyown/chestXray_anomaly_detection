import torch
import torchvision

modelFastRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True
)
torch.save(modelFastRCNN.state_dict(), 'fasterrcnn_weights.pth')