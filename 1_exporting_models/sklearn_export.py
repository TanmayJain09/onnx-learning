from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
from pathlib import Path

output_dir = Path(__file__).resolve().parent
output_dir = output_dir.parent/"assets"/"sample_models"
output_dir.mkdir(parents=True,exist_ok=True)

output_file_path = output_dir/"sklearn_exported.onnx"

# Train a model
X, y = make_classification(n_samples=1000, n_features=20)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# The initial_types argument tells skl2onnx the input shape and dtype
# None means the batch size is dynamic
initial_type = [("float_input", FloatTensorType([None, 20]))]

onnx_model = convert_sklearn(clf, initial_types= initial_type , target_opset=17)

# Save it
with open(output_file_path, "wb") as f:
    f.write(onnx_model.SerializeToString())