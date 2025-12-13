import gradio as gr
import torch
from src.config import CONFIG, PATHS
from src.dataset import get_transforms
from src.model import build_model

# Constants
LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
WEIGHTS_PATH = PATHS["weights_dir"] / "effnetb3_fold0.pth"

# Load Model
model = build_model(len(LABELS), CONFIG["device"])
if WEIGHTS_PATH.exists():
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=CONFIG["device"]))
else:
    print("Warning: Weights not found. App will use random weights.")

model.eval()
_, val_tf = get_transforms(CONFIG["img_size"])

def predict(image):
    if image is None: return None
    img_t = val_tf(image.convert("RGB")).unsqueeze(0).to(CONFIG["device"])
    
    with torch.no_grad():
        probs = torch.softmax(model(img_t), dim=1).cpu().numpy()[0]
        
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

demo = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(num_top_classes=3),
    title="Skin Lesion Classifier (B3)"
)

if __name__ == "__main__":
    demo.launch()