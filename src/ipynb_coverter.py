import nbconvert

notebook_path = (
    "C:/Users/saiprasath/ML-Surface-Crack-Detection/notebooks/Preprocessing.ipynb"
)

output_path = (
    "C:/Users/saiprasath/ML-Surface-Crack-Detection/src/modeling/preprocessing.py"
)

# Convert the notebook
exporter = nbconvert.exporters.PythonExporter()
content, _ = exporter.from_filename(notebook_path)

# Save as .py file in the desired location
with open(output_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Notebook converted and saved to {output_path}")
