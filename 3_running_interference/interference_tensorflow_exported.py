import onnxruntime as ort
import numpy as np
from pathlib import Path 

output_dir = Path(__file__).resolve().parent
output_dir = output_dir.parent / "assets" / "sample_models" / "tensorflow_exported.onnx"

# CPU-only session (fix provider issue)
session = ort.InferenceSession(
    output_dir,
    providers=["CPUExecutionProvider"]
)

# Print active providers
print(session.get_providers())

# Create input tensor (IMPORTANT: match expected format)
img = np.random.randn(1, 224, 224, 3).astype(np.float32)

# Run inference (FIXED input name)
outputs = session.run(None, {"input": img})

logits = outputs[0]
predicted_class = np.argmax(logits, axis=1)

print(f"Predicted class: {predicted_class}")