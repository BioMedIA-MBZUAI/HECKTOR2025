import torch
import torch.nn as nn
from monai.networks.nets import resnet18

class MultiModalResNet(nn.Module):
    def __init__(self, clin_feat_dim, num_classes=2):
        super().__init__()
        # 3D ResNet18 backbone for CT+PET
        self.img_model = resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=2,
        )
        self.img_model.fc = nn.Identity()

        # MLP for clinical data
        self.clin_model = nn.Sequential(
            nn.Linear(clin_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # fusion + classification
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_clin):
        f_img = self.img_model(x_img)    
        f_clin = self.clin_model(x_clin)   
        f = torch.cat([f_img, f_clin], dim=1)
        return self.classifier(f)