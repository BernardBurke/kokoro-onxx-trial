import asyncio
import soundfile as sf
import numpy as np
import argparse
import os
import sys
import subprocess
import re
import onnxruntime as rt

from kokoro_onnx import SAMPLE_RATE, Kokoro

# --- CONFIGURATION ---
# The Safe Limit: 350 chars is roughly ~80-90 tokens. 
# The Hard Limit is 510, so this gives us a huge safety margin.
MAX_CHAR_LIMIT = 350 

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

def smart_split(text, limit=MAX_CHAR_LIMIT):
    """
    Splits text aggressively if it exceeds the limit.
    Priority: Newlines -> .?! -> ;: -> , -> Space -> Hard Cut
    """
    text = text.strip()
    if not text:
        return []
    
    # If it fits, return it
    if len(text) <= limit:
        return [text]
    
    # Strategy 1: Split by sentence (.?!)
    # Lookbehind ensures we keep the punctuation
    parts = re.split(r'(?<=[.?!])\s+', text)
    if len(parts) > 1:
        # Check if splitting helped
        if all(len(p) <= limit for p in parts):
            return parts
        # If some parts are still too big, recurse
        final_parts = []
        for p in parts:
            final_parts.extend(smart_split(p, limit))
        return final_parts
        
    # Strategy 2: Split by sub-clauses (; or :)
    parts = re.split(r'(?<=[;:])\s+', text)
    if len(parts) > 1:
        final_parts = []
        for p in parts:
            final_parts.extend(smart_split(p, limit))
        return final_parts

    # Strategy 3: Split by comma
    parts = re.split(r'(?<=,)\s+', text)
    if len(parts) > 1:
        final_parts = []
        for p in parts:
            final_parts.extend(smart_split(p, limit))
        return final_parts

    # Strategy 4: Split by space (last resort) to ensure safety
    words = text.split(' ')
    current_chunk = []
    current_len = 0
    chunks = []
    
    for word in words:
        if current_len + len(word) + 1 > limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = len(word)
        else:
            current_chunk.append(word)
            current_len += len(word) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

async def main(input_file, voice_name, output_m4a_path):
    # --- Kokoro Initialization ---
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    os.environ.pop('ONNX_EXECUTION_PROVIDERS', None) 

    print("Initializing Kokoro...")
    try:
        # Use v1.0 files as confirmed
        session = rt.InferenceSession("kokoro-v1.0.onnx", providers=providers)
        kokoro = Kokoro(session=session, voices_path="voices-v1.0.bin")
        print("INFO: Kokoro initialized (Session method).")
    except Exception:
        print("WARNING: Fallback to basic init...")
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

    # --- READ AND CHOP THE FILE ---
    print(f"Reading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read all lines first
        raw_lines = f.readlines()

    safe_chunks = []
    for line in raw_lines:
        # Pass every line through the smart splitter
        # This fixes the "Frankenstein sentences" that have no periods
        safe_chunks.extend(smart_split(line))

    # Filter empty chunks
    safe_chunks = [c for c in safe_chunks if c.strip()]
    
    total_chunks = len(safe_chunks)
    print(f"Safety Split Complete: Processing {total_chunks} safe chunks.")

    # --- GENERATE AUDIO ---
    all_audio = []
    sample_rate = 24000
    
    print(f"Starting generation with voice: {voice_name}")
    
    # We loop MANUALLY. We do NOT use create_stream on the whole text.
    # This guarantees the model never sees a string > MAX_CHAR_LIMIT.
    for i, chunk in enumerate(safe_chunks):
        pct = int((i / total_chunks) * 100)
        sys.stdout.write(f"\rProgress: [{pct:3d}%] Chunk {i+1}/{total_chunks}")
        sys.stdout.flush()
        
        try:
            # We use create() for single chunks, it's safer than stream for this approach
            audio, _ = kokoro.create(
                chunk, 
                voice=voice_name, 
                speed=1.0, 
                lang="en-us"
            )
            all_audio.append(audio)
            
            # Tiny pause between chunks for natural flow
            # If the chunk ended with sentence punctuation, add 0.2s
            # If it was a mid-sentence chop (comma), add 0.1s
            if chunk.strip()[-1] in '.?!':
                all_audio.append(np.zeros(int(0.2 * sample_rate)))
            else:
                all_audio.append(np.zeros(int(0.05 * sample_rate)))
                
        except Exception as e:
            print(f"\nError on chunk '{chunk[:20]}...': {e}")
            # Don't exit, just skip this bad chunk
            continue

    print(f"\nSynthesis Complete!")

    if not all_audio:
        print("Error: No audio generated.")
        sys.exit(1)

    # --- SAVE ---
    print("Concatenating and converting...")
    final_audio = np.concatenate(all_audio)
    pcm_s16le = (final_audio * 32767).astype(np.int16)

    ffmpeg_cmd = [
        'ffmpeg', '-f', 's16le', '-ar', str(sample_rate), '-ac', '1',
        '-i', 'pipe:0', '-c:a', 'aac', '-vn', '-y', output_m4a_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, input=pcm_s16le.tobytes(), check=True, capture_output=True)
        print(f"Success! Saved to: {output_m4a_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr.decode()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kokoro_tts_safe.py <input_file> [voice]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    voice = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VOICE
    
    if not os.path.exists(input_file):
        print("File not found.")
        sys.exit(1)

    output = f"{os.path.splitext(input_file)[0]}_{voice}.m4a"
    asyncio.run(main(input_file, voice, output))
