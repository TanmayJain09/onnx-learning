import tkinter as tk
from tkinter import filedialog
from pathlib import Path 
import os
import onnx 
import json
from collections import Counter

#defining output directory
parent_dir = Path(__file__).resolve().parent.parent
output_dir = parent_dir / "assets" / "models_report"
output_dir.mkdir(parents=True, exist_ok=True)

#taking the file input from the user 
def get_onnx_file () : 
    
    #hide the root window
    root = tk.Tk()
    root.withdraw()

    #get the file from user
    file_path = filedialog.askopenfilename(
        title= "Select ONNX File Only",
        filetypes=[("ONNX files","*.onnx")]
    )

    if not file_path : 
        return None, None

    file_name = os.path.basename(file_path)
    name_without_extension = os.path.splitext(file_name)[0]

    return file_path,name_without_extension


#shape parser function
def shape_parser(tensor) : 
    shape = []
    for d in tensor.type.tensor_type.shape.dim : 
        if d.dim_value > 0 : 
            shape.append(d.dim_value)
        elif d.dim_param : 
            shape.append(d.dim_param)
        else : 
            shape.append("dynamic")
    return shape

#inspection function
def inspect_onnx (model_path, model_name) : 

    model = onnx.load(model_path)
    graph = model.graph

    report = {
        "model_name" : model_name,
        "num_nodes" : len(graph.node),
        "inputs" : [],
        "outputs" : [],
        "op_counts" : {},
        "unique_ops" : [],
        "warnings" : [],
        "complexity_score" : 0 
    }

    #input for the graph
    for inp in graph.input : 
        report["inputs"].append({
            "name" : inp.name,
            "shape" : shape_parser(inp),
            "dtype" : inp.type.tensor_type.elem_type
        })

    #output of the graph
    for out in graph.output:
        report["outputs"].append(out.name)

    #operators
    ops = [n.op_type for n in graph.node]
    op_counts = Counter(ops)

    report["op_counts"] = dict(op_counts)
    report["unique_ops"] = list(op_counts.keys())

    #complexity score
    score = len(graph.node)
    report["complexity_score"] = score

    #debugging warning (model size)
    if score < 20:
        report["warnings"].append("Very small graph — possibly incomplete export")

    if score > 1000:
        report["warnings"].append("Very large graph — may be heavy for inference")

    # unusual ops detection
    unusual_ops = set(ops) - {
        "Conv", "Relu", "BatchNormalization",
        "MaxPool", "Add", "Gemm", "MatMul",
        "Softmax", "Flatten", "Concat"
    }

    if unusual_ops:
        report["warnings"].append(f"Unusual ops detected: {list(unusual_ops)}")

    # dominance check
    if "Conv" in op_counts and op_counts["Conv"] / len(ops) > 0.6:
        report["warnings"].append("Conv-heavy model")

    # output file name and directory
    
    # saving json
    output_file_json_name = "report_"+model_name+".json"
    output_file_json_dir = output_dir/output_file_json_name
    with open(output_file_json_dir, "w") as f:
        json.dump(report, f, indent=4)

    # text report
    output_file_txt_report_name = "report_"+model_name+".txt"
    output_file_txt_report_dir = output_dir/output_file_txt_report_name
    with open(output_file_txt_report_dir, "w") as f:
        f.write("ONNX MODEL REPORT\n")
        f.write("==================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Nodes: {score}\n\n")

        f.write("INPUTS:\n")
        for i in report["inputs"]:
            f.write(f"- {i['name']} | {i['shape']} | {i['dtype']}\n")

        f.write("\nOUTPUTS:\n")
        for o in report["outputs"]:
            f.write(f"- {o}\n")

        f.write("\nOPERATORS:\n")
        for k, v in op_counts.most_common():
            f.write(f"- {k}: {v}\n")

        f.write("\nUNIQUE OPS:\n")
        f.write(", ".join(report["unique_ops"]))


def main() : 
    file_path , file_name = get_onnx_file()
    inspect_onnx(file_path,file_name)

if __name__ == "__main__" : 
    main()