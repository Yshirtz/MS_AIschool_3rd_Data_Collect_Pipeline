import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class ResNet50TripletNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # weights=None: 초기화 시 랜덤 가중치 사용 (이후 load_state_dict로 덮어씌움)
        backbone = models.resnet50(weights=None) 
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, embedding_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# ResNet 전용 이미지 전처리 파이프라인
resnet_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])