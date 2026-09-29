"""
PyTorch Grad-CAM (Gradient-Weighted Class Activation Mapping) Generator for DenseNet121.
Computes gradient localization maps for the last convolutional layer.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, meta_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        logits, _ = self.model(input_tensor, meta_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # Target score for backprop
        score = logits[0, target_class]
        score.backward()
        
        # Global Average Pooling on gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activation channels
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)
        
        return heatmap.cpu().detach().numpy()

if __name__ == '__main__':
    print("Grad-CAM module compiled successfully.")
