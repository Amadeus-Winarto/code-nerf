import timm
import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, model_name="resnet34"):
        super().__init__()
        if model_name == "resnet34":
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(2, 3, 4),
            )

            self.cls1 = nn.Sequential(
                ResNetBlock(128, 256, 2),
                ResNetBlock(256, 256),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.cls2 = nn.Sequential(
                ResNetBlock(256, 256),
                ResNetBlock(256, 256),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.cls3 = nn.Sequential(
                ResNetBlock(512, 256),
                ResNetBlock(256, 256),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

            self.final_cls = nn.Sequential(
                nn.Linear(256, 256), nn.Linear(256, 128), nn.Linear(128, 1)
            )

        else:
            raise NotImplementedError(f"{model_name} is not implemented")

    def extract_features(self, x, enable_grad=False):
        if enable_grad:
            features = self.model(x)
        else:
            with torch.no_grad():
                features = self.model(x)

        for f in features:
            f.requires_grad_(True)
        return features

    def get_logits(self, x):
        x1, x2, x3 = x
        x1 = self.cls1(x1)
        x2 = self.cls2(x2)
        x3 = self.cls3(x3)

        return self.final_cls(x1 + x2 + x3)

    def forward(self, x):
        features = self.extract_features(x)
        return features, self.get_logits(features)


def r1_penalty(logits, features, gamma=0.2):
    r1_grads = torch.autograd.grad(
        outputs=[logits.sum()], inputs=features, create_graph=True
    )[0]
    r1_penalty = r1_grads.square().sum(dim=(1, 2, 3))

    return gamma / 2 * r1_penalty.mean()


if __name__ == "__main__":
    model = Discriminator()
    gamma = 0.2

    x = torch.randn(32, 3, 224, 224)
    features = model.extract_features(x)
    logits = model.get_logits(features)

    # Feature R1 regularization
    loss = r1_penalty(logits, features, gamma=gamma)
    loss.backward()
    print("Loss: ", loss.item())

    
