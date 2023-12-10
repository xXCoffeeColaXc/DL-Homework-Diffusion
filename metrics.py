import torch
from torch import Tensor
from torchvision import models, transforms
from torchvision.models.inception import Inception_V3_Weights
from torchmetrics import MeanMetric
from torchmetrics.metric import Metric
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Union, List

class KID(Metric):
    def __init__(self, device, target_layer_name: str = "avgpool"):
        super().__init__()
        self.config_device = device
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
        self.inception.eval()

        # InceptionV3 input  size should be (299, 299)
        self.target_size = (299, 299)
        self.target_layer_names = ["avgpool"]
        if not (target_layer_name in self.target_layer_names):
            raise ValueError("Argument `target_layer_name` is invalid")
        self.target_layer_name = target_layer_name
        self.activations = self.register_hooks()

        self.kid_mean = MeanMetric().to(device)

        self.preprocess_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def register_hooks(self):
        activations = {}

        def hook_fn(module, input, output, key):
            activations[key] = output.detach()

        for name, layer in self.get_target_layers().items():
            layer.register_forward_hook(partial(hook_fn, key=name))

        return activations

    def get_target_layers(self):
        return {name: layer for name, layer in self.inception.named_children() if name in self.target_layer_names}

    def extract_features(self, input_batch: Tensor):
        output = self.inception(input_batch)
        activation = self.activations[self.target_layer_name]
        if self.target_layer_name == "avgpool":
            activation = torch.flatten(activation, 1)

        return activation

    def polynomial_kernel(self, features_1: Tensor, features_2: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0):
        if gamma is None:
            gamma = 1.0 / features_1.shape[1]
        return (features_1 @ features_2.T * gamma + coef) ** degree

    def update(self, real_images: Tensor, generated_images : Tensor):
        real_input = self.preprocess_transform(real_images)
        generated_input = self.preprocess_transform(generated_images)
        real_features = self.extract_features(real_input)
        generated_features = self.extract_features(generated_input)

        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        batch_size = real_features.size(0)
        batch_size_f = float(batch_size)

        mean_kernel_real = (kernel_real.sum(dim=-1) - torch.diag(kernel_real)).sum() / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_generated = (kernel_generated.sum(dim=-1) - torch.diag(kernel_generated)).sum() / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = torch.mean(kernel_cross)

        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        print("MY kid value: ", kid)

        self.kid_mean.update(kid)

    def compute(self):
        return self.kid_mean.compute()

    def reset(self):
        self.kid_mean.reset()

