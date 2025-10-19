import torch  # <-- ADD THIS LINE
import onnxruntime
from kokoro_onnx import Kokoro 
import soundfile as sf
import time

# --- 1. Settings ---
text_to_say = "This is a new test. This time, it should use the GPU and successfully save the file."
output_filename = "onnx_gpu_output.wav"

# Model files in the same folder
model_path = "kokoro-v1.0.onnx"
voices_path = "voices-v1.0.bin"


# This is how you tell onnxruntime to use CUDA
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# --- 2. Initialize ---
print(f"Loading ONNX session for: {model_path}")
print(f"Using providers: {providers}")
session = onnxruntime.InferenceSession(model_path, providers=providers)

print(f"Loading Kokoro engine with voices from: {voices_path}")
kokoro = Kokoro.from_session(session, voices_path)

print("Engine initialized.")

# --- 3. Generate Speech ---
print(f"Generating speech for: '{text_to_say}'")
start_time = time.time()

# The method is .create()
samples, sample_rate = kokoro.create(
    text=text_to_say, 
    voice="af_sarah"
)

end_time = time.time()
print(f"Speech generated in {end_time - start_time:.2f} seconds.")

# --- 4. Save the Audio ---
# THE FIX IS HERE: Changed 'audio' to 'samples'
sf.write(output_filename, samples, sample_rate) 
print(f"Test complete. Audio saved to '{output_filename}'.")
