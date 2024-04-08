import torch.nn as nn
import torch.nn.functional as F


class AuxClassifier(nn.Module):
    def __init__(self, inplanes,class_num=10, widen=1, feature_dim=128):
        super(AuxClassifier, self).__init__()

        assert inplanes in [16, 32, 64]

        self.feature_dim = feature_dim

        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = class_num

        self.head = nn.Sequential(
                    nn.Conv2d(inplanes,feature_dim,1,1,0,bias=False),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim,feature_dim,1,1,0,bias=False),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim, feature_dim, 1, 1, 0,bias=False),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(feature_dim,feature_dim * 4),
                    nn.BatchNorm1d(feature_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim * 4,feature_dim * 4),
                    nn.BatchNorm1d(feature_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim * 4, feature_dim * 4),
                    nn.BatchNorm1d(feature_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim * 4,class_num)
        )

    def forward(self, x, target):
        x = F.adaptive_avg_pool2d(x,(8,8))
        features = self.head(x)
        loss = self.criterion(features, target)
        return loss