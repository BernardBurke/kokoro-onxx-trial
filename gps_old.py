import asyncio
import soundfile as sf
import numpy as np
import argparse
import os
import sys
import subprocess
import onnxruntime as rt # <-- NEW IMPORT for manual session

from kokoro_onnx import SAMPLE_RATE, Kokoro

# --- List of Valid Voices ---
VALID_VOICES = [
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore',
    'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam',
    'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx',
    'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora', 'em_alex',
    'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi',
    'if_sara', 'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi',
    'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex', 'pm_santa', 'zf_xiaobei',
    'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi'
]
DEFAULT_VOICE = "af_nicole"

# --- CONFIGURATION FOR PROGRESS ESTIMATE ---
CHARS_PER_CHUNK_ESTIMATE = 100 
SAMPLES_PER_CHUNK = 2400 

async def main(input_text, voice_name, output_m4a_path):
    # --- Kokoro Initialization (with manual ONNX Session) ---
    
    # 1. Define the execution providers, prioritizing CUDA
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    # Clean up any residual environment variable setting from previous attempts
    os.environ.pop('ONNX_EXECUTION_PROVIDERS', None) 

    try:
        # 2. Try to create the ONNX Runtime InferenceSession directly with providers
        session = rt.InferenceSession("kokoro-v1.0.onnx", providers=providers)
        
        # 3. Initialize Kokoro by passing the created session and the voices path
        # NOTE: This assumes Kokoro supports initializing with a 'session' argument.
        kokoro = Kokoro(
            session=session, 
            voices_path="voices-v1.0.bin"
        )
        print("INFO: Kokoro initialized using a manually configured ONNX session (GPU preferred).")

    except TypeError as e:
        # Catches the error if Kokoro doesn't accept the 'session' argument
        print(f"WARNING: Kokoro constructor does not accept 'session' argument. Falling back to environment variable method...")
        
        # Try the environment variable method as the next best option
        os.environ['ONNX_EXECUTION_PROVIDERS'] = 'CUDAExecutionProvider'
        try:
            kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            print("INFO: Kokoro initialized using environment variable (GPU preferred).")
        except Exception as inner_e:
            print(f"ERROR: Fallback initialization failed: {inner_e}")
            sys.exit(1)
            
    except Exception as e:
        # Catches other initialization failures (e.g., model file not found)
        print(f"ERROR: Initialization failed: {e}. Falling back to default CPU setup.")
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin") 
        os.environ.pop('ONNX_EXECUTION_PROVIDERS', None)
    
    # --- Check for successful initialization before proceeding ---
    if not 'kokoro' in locals() and not 'kokoro' in globals():
        print("FATAL: Kokoro object could not be initialized by any method.")
        sys.exit(1)

    # --- Pre-calculate Estimated Total Chunks ---
    text_length = len(input_text)
    estimated_total_chunks = max(1, (text_length // CHARS_PER_CHUNK_ESTIMATE) + 10)
    
    print(f"Generating speech with voice: {voice_name}")
    print(f"Estimated total chunks to process: {estimated_total_chunks}")
    
    stream = kokoro.create_stream(
        input_text,
        voice=voice_name,
        speed=1.0,
        lang="en-us",
    )

    # --- Collect Audio Chunks and Display Progress ---
    all_samples_list = []
    chunk_count = 0
    total_samples = 0
    
    sys.stdout.write("Progress: [ 0% ]")
    sys.stdout.flush() 

    async for samples, _ in stream:
        chunk_count += 1
        total_samples += samples.shape[0]
        all_samples_list.append(samples)
        
        # Calculate progress using the estimated total chunks
        progress_percent = int((chunk_count / estimated_total_chunks) * 100)
        
        # Dynamically update the estimate if synthesis takes longer
        if chunk_count > estimated_total_chunks:
             estimated_total_chunks = int(chunk_count * 1.20)

        # Print the progress indicator, using \r to reset the line
        sys.stdout.write(f"\rProgress: [{progress_percent:3d}%] Chunks: {chunk_count}/{estimated_total_chunks} | Duration: {total_samples / SAMPLE_RATE:.1f}s")
        sys.stdout.flush()

    # Final line printout to confirm completion
    sys.stdout.write(f"\rSynthesis Complete: [100%] Chunks: {chunk_count} | Duration: {total_samples / SAMPLE_RATE:.1f}s\n")
    sys.stdout.flush()


    if not all_samples_list:
        print("Error: No audio data received from Kokoro stream.")
        sys.exit(1)

    # --- Concatenate and Convert Audio Data ---
    print("Concatenating audio chunks...")
    concatenated_samples = np.concatenate(all_samples_list)

    # Convert float samples (-1.0 to 1.0) to 16-bit PCM for ffmpeg
    print("Converting audio data format...")
    pcm_s16le = (concatenated_samples * 32767).astype(np.int16)

    # --- Convert to M4A using ffmpeg ---
    print(f"Converting to M4A: {output_m4a_path}")
    ffmpeg_command = [
        'ffmpeg',
        '-f', 's16le',        # Input format: signed 16-bit little-endian PCM
        '-ar', str(SAMPLE_RATE), # Input sample rate
        '-ac', '1',           # Input channels (mono)
        '-i', 'pipe:0',       # Read PCM data from stdin
        '-c:a', 'aac',        # Output codec: AAC (standard for M4A)
        '-vn',                # No video output
        '-y',                 # Overwrite output file if it exists
        output_m4a_path
    ]

    try:
        process = subprocess.run(
            ffmpeg_command,
            input=pcm_s16le.tobytes(),
            check=True,              
            capture_output=True       
        )
        print("FFmpeg conversion successful.")
    except FileNotFoundError:
        print("\nError: 'ffmpeg' command not found.")
        print("Please ensure ffmpeg is installed and in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError during ffmpeg conversion (return code {e.returncode}):")
        print(e.stderr.decode())
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech from a text file using Kokoro TTS and save as M4A.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument(
        "voice",
        nargs='?', 
        default=DEFAULT_VOICE,
        help=f"Voice name to use (default: {DEFAULT_VOICE}). Choose from available voices."
    )

    args = parser.parse_args()

    # --- Validate Input File ---
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # --- Validate Voice ---
    if args.voice not in VALID_VOICES:
        print(f"Error: Invalid voice name '{args.voice}'.")
        print("Available voices are:")
        for v in VALID_VOICES:
            print(f"- {v}")
        sys.exit(1)

    # --- Read Input Text ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        if not text_content.strip():
             print(f"Error: Input file '{args.input_file}' is empty.")
             sys.exit(1)
    except Exception as e:
        print(f"Error reading input file '{args.input_file}': {e}")
        sys.exit(1)


    # --- Determine Output Filename ---
    base_name, _ = os.path.splitext(args.input_file)
    output_path = f"{base_name}_{args.voice}.m4a"

    # --- Run Main Function ---
    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_path}")
    asyncio.run(main(text_content, args.voice, output_path))

    print("Script finished.")
