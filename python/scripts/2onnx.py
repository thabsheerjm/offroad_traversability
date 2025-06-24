import torch, torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from src.model import DeepLabHead

def pth2onnx(pth_model_file, onnx_model_file):
    model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=False)
    model.classifier = DeepLabHead(960, 1) 
    model.load_state_dict(torch.load(pth_model_file, map_location="cpu"))
    model.eval()

    dummy_in = torch.randn(1,3,520,520)

    torch.onnx.export(
        model, dummy_in, 
        onnx_model_file,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}},
        opset_version=17 
    )
    return True

if __name__ == '__main__':
    
    pth_file = "models/finetuned/deeplabv3_mnv3_best.pth"
    onnx_file = "models/finetuned/deeplabv3_mnv3_finetuned.onnx"

    ret = pth2onnx(pth_file,onnx_file)

    if ret:
        print("Pytorch model is converted to Onnx")
    else:
        print("failed to convert pth to onnx!!")
