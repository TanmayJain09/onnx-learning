import onnxruntime as ort
import numpy as np
import time
import json
import shutil
import os
from pathlib import Path

# setting up directories
parent_dir = Path(__file__).resolve().parent.parent / "assets"
base_input_model = parent_dir / "sample_models" / "tensorflow_exported.onnx"
opt_dir = parent_dir / "optimise"

opt_models = opt_dir / "models"
os.makedirs(opt_models,exist_ok=True)

opt_profiles = opt_dir / "profiles"
os.makedirs(opt_profiles,exist_ok=True)

opt_results = opt_dir / "results"
os.makedirs(opt_results,exist_ok=True)

opt_reports = opt_dir / "reports"
os.makedirs(opt_reports,exist_ok=True)

# input tensor (fixed for fair comparison)
x = np.random.randn(1, 224, 224, 3).astype(np.float32)

#run function
def run(model_path, name, session_options) : 
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )
    
    input_name = session.get_inputs()[0].name

    #warmup
    for _ in range(5):
        session.run(None, {input_name : x})
    
    # timing
    times=[]
    output=None

    for _ in range(20) : 
        start = time.time()
        output = session.run(None,{input_name : x})
        end = time.time()
        times.append(end-start)
    
    #save output
    np.save(f"results/{name}.npy",output[0])

    return{
        "avg_latency" : float(np.mean(times)),
        "min_latency" : float(np.min(times)),
        "max_latency" : float(np.max(times))
    }

def raw():

    raw_model_path = opt_models / "raw.onnx"

    # copy base model into your controlled directory
    shutil.copy(base_input_model, raw_model_path)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    so.enable_profiling = True
    so.profile_file_prefix = str(opt_profiles / "raw")

    return run(raw_model_path, "raw", so)


#main function
def main():

    results = {}

    results["raw"] = raw()

    #saving report
    with open("report/benchmark.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n====================")
    print("BENCHMARK COMPLETE")
    print("====================")

    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()