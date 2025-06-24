import os, cv2, torch,numpy as np, json, torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch.nn import functional as F 
from pathlib import Path 
from src.model import DeepLabHead

def test_config(config_path="src/config/config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        return config
    
def load_model(checkpoint_path, device="cpu"):
    model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=False)
    model.classifier = DeepLabHead(960, 1) 
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_image(image, size=(520, 520)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # [1, 3, H, W]
    return tensor, image 

def overlay_mask(image_rgb, mask, alpha=0.5):
    mask = (mask > 0.5).astype(np.uint8)
    overlay = np.zeros_like(image_rgb)
    overlay[mask == 1] = [0, 255, 0]   # Green for traversable
    overlay[mask == 0] = [0, 0, 255]   # Blue for non-traversable
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    return blended

def run_inference_on_folder(model, input_folder, output_folder, device="cpu"):
    os.makedirs(output_folder, exist_ok=True)
    input_paths = list(Path(input_folder).glob("*.png"))

    for path in input_paths:
        image = cv2.imread(str(path))
        input_tensor, original_rgb = preprocess_image(image)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)['out']
            mask = torch.sigmoid(output).squeeze().cpu().numpy()  

        original_rgb = cv2.resize(original_rgb, mask.shape[::-1])  
        overlay = overlay_mask(original_rgb, mask)
        save_path = os.path.join(output_folder, path.stem + "_overlay.png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")
    
def inference_test():
    config = test_config()
    checkpoint = config["training"]["checkpoint_path"]
    test_dir = config["test"]["test_dir"]
    save_dir = config["test"]["save_test_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint).to(device)
    run_inference_on_folder(model, test_dir, save_dir, device)
    return True


if __name__ == '__main__':
    ret = inference_test()
    if ret:
        print("inference test was successfull and results are saved!")
    else:
        print("Inference failure!")