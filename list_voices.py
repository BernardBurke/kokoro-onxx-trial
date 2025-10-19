import onnxruntime
from kokoro_onnx import Kokoro
import os

# --- 1. Settings ---
# Assumes models are in the same folder as the script
models_dir = "."
model_path = os.path.join(models_dir, "kokoro-v1.0.onnx")
voices_path = os.path.join(models_dir, "voices-v1.0.bin")

# --- 2. Initialize ---
# Use CPU provider for just listing voices, it's faster to load
providers = ["CPUExecutionProvider"] 
session = onnxruntime.InferenceSession(model_path, providers=providers)
kokoro = Kokoro.from_session(session, voices_path)

# --- 3. List Voices ---
print("Available voices:")
# The kokoro.voices attribute is a dictionary; we want the keys (names)
for voice_name in kokoro.voices.keys():
    print(f"- {voice_name}")

print("\nEngine initialized successfully.")
