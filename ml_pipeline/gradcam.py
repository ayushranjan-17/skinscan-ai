"""
PyTorch Grad-CAM (Gradient-Weighted Class Activation Mapping) Generator for DenseNet121.
Computes gradient localization maps for the last convolutional layer.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        if target_layer is None:
            # Default to conv5_block16_2_conv (features.denseblock4.denselayer16.conv2)
            try:
                self.target_layer = model.vision_backbone.features.denseblock4.denselayer16.conv2
            except AttributeError:
                self.target_layer = getattr(model.vision_backbone.features, "denseblock4")
        else:
            self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [h_fwd, h_bwd]

    def remove_hooks(self):
        """Cleanly remove registered forward and backward hooks to prevent memory leaks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_heatmap(self, input_tensor, meta_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        # Ensure gradient tracking on input
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)
        
        # Forward pass
        logits, _ = self.model(input_tensor, meta_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # Target score for backprop
        score = logits[0, target_class]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global Average Pooling on gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activation channels without in-place mutation
        activations = self.activations.clone().detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val + 1e-8
        
        return heatmap.cpu().detach().numpy()

if __name__ == '__main__':
    print("Grad-CAM module compiled successfully.")
