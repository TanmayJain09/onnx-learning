import tensorflow as tf
import tf2onnx
from pathlib import Path

output_dir = Path(__file__).resolve().parent
output_dir = output_dir.parent/"assets"/"sample_models"
output_dir.mkdir(parents=True,exist_ok=True)

output_file_path = output_dir/"tensorflow_exported.onnx"

#loading keras model
model = tf.keras.applications.MobileNetV2()

#convert to onnx
input_spec = (tf.TensorSpec((None, 224, 224, 3),tf.float32,name="input"),)

model_proto,_ = tf2onnx.convert.from_keras(
    model,
    input_signature = input_spec,
    opset = 17,
    output_path = output_file_path,
)

print("Model Exported at : ",output_file_path)