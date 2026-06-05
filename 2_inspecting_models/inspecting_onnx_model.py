from pathlib import Path 

parent_dir = Path(__file__).resolve().parent.parent
output_dir = parent_dir / "assets" / "models_report"
output_dir.mkdir(parents=True, exist_ok=True)
