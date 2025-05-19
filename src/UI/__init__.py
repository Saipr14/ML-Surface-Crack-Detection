import sys
import os

# Get absolute path to the modeling folder
modeling_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "modeling")
)
sys.path.insert(0, modeling_path)
