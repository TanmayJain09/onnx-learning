import tkinter as tk
from tkinter import filedialog
from pathlib import Path 
import os

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

def main() : 
    file_path , file_name = get_onnx_file()
    print("File Path :",file_path)
    print("File Name :",file_name)

if __name__ == "__main__" : 
    main()