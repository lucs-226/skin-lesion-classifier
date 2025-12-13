import cv2
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        return np.maximum(heatmap, 0)

def visualize_sample_with_confidence(model, dataset, device, classes_map):
    idx = random.randint(0, len(dataset)-1)
    img_tensor, label_idx = dataset[idx]
    
    class_names = sorted(classes_map.keys())
    true_label_name = class_names[label_idx]
    
    model.eval()
    input_tensor = img_tensor.unsqueeze(0).to(device)
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    with torch.enable_grad():
        heatmap = cam(input_tensor, label_idx)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    rgb_img = inv_normalize(img_tensor).permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip(rgb_img, 0, 1)

    heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
    if np.max(heatmap_resized) != 0: 
        heatmap_resized /= np.max(heatmap_resized)
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = np.float32(heatmap_colored) / 255
    
    overlay = np.clip(heatmap_colored * 0.4 + rgb_img * 0.6, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Original: {true_label_name}")
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title("GradCAM Focus")
    axes[1].axis('off')
    
    sorted_idxs = np.argsort(probs)
    sorted_probs = probs[sorted_idxs]
    sorted_names = [class_names[i] for i in sorted_idxs]
    
    colors = ['gray'] * len(sorted_names)
    if true_label_name in sorted_names:
        colors[sorted_names.index(true_label_name)] = 'green'
        
    y_pos = np.arange(len(sorted_names))
    axes[2].barh(y_pos, sorted_probs, color=colors)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(sorted_names)
    axes[2].set_title("Confidence")
    
    for i, v in enumerate(sorted_probs):
        axes[2].text(v + 0.01, i, f"{v:.1%}", va='center')

    plt.tight_layout()
    plt.show()