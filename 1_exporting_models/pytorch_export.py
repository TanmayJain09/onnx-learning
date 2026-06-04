import torch
import torchvision
from pathlib import Path 

output_dir = Path(__file__).resolve().parent
output_dir = output_dir.parent / "assets" / "sample_models"
output_dir.mkdir(parents=True, exist_ok=True)

output_file_path = output_dir / "pytorch_exported.onnx"

model = torchvision.models.resnet50(pretrained=True)
model.eval()

dummy_input = torch.randn(1,3,224,224)

torch.onnx.export(
    model,
    dummy_input,
    output_file_path,
    opset_version=17,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image" : {0:"batch_size"},
        "logits" : {0:"batch_size"}
    }
)

print("Model Exported at : ",output_file_path)