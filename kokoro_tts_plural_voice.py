import asyncio
import numpy as np
import argparse
import os
import sys
import subprocess
import re
import io # NEW: Used for handling in-memory audio segments

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

def segment_text(text, default_voice):
    """
    Splits the input text into a list of (voice_name, text_segment) tuples
    based on the presence of <voice=name> tags.
    """
    # 1. Pre-process the text to ensure tags are recognized correctly
    # Replace newlines with spaces, but keep the tags
    text = text.strip()
    
    # Use a regex that splits the text, keeping the delimiter (the voice tag)
    # The pattern is: (A voice tag followed by optional whitespace)
    # The parentheses around the tag ensure re.split keeps the delimiter in the resulting list
    segments = re.split(r'(<voice=[a-z0-9_]+>\s*)', text, flags=re.IGNORECASE)
    
    # Filter out empty strings that result from the split
    segments = [s for s in segments if s.strip()]

    result = []
    current_voice = default_voice
    
    # If the text does not start with a tag, the first segment is the text itself
    if not segments[0].lower().startswith('<voice='):
        result.append((current_voice, segments[0].strip()))
        start_index = 1
    else:
        start_index = 0

    # 2. Process the remaining segments (which will alternate between tag and text)
    for i in range(start_index, len(segments)):
        segment = segments[i].strip()
        
        if segment.lower().startswith('<voice='):
            # This segment is a tag, extract the new voice name
            match = re.match(r'<voice=([a-z0-9_]+)>', segment, re.IGNORECASE)
            if match:
                current_voice = match.group(1).lower()
            # If a tag appears without text following it (e.g., at the end of the file), ignore it.
        else:
            # This segment is the text content
            if segment:
                result.append((current_voice, segment))

    return result

async def main(input_text, voice_name, output_m4a_path):
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

    # --- NEW: Segment the Text ---
    # The default_voice passed here is the command-line argument (af_nicole)
    segments = segment_text(input_text, voice_name)
    
    if not segments:
        print("Error: Input text contains no readable content after segmentation.")
        sys.exit(1)

    all_samples_list = []
    total_chunks = 0
    
    print(f"Text segmented into {len(segments)} blocks for processing.")

    # --- NEW: Process Each Segment Individually ---
    for i, (voice, text) in enumerate(segments):
        if voice not in VALID_VOICES:
            print(f"\nCRITICAL ERROR: Voice '{voice}' (from segment {i+1}) is not a valid voice!")
            sys.exit(1)
            
        print(f"\nProcessing segment {i+1}/{len(segments)} with voice: {voice}")
        print(f"Text: \"{text[:70]}...\"")
        
        # Call create_stream with a single, explicit voice name for this segment
        stream = kokoro.create_stream(
            text,
            voice=voice, # Pass the voice name *explicitly*
            speed=1.0,
            lang="en-us",
        )

        # Collect audio chunks for this segment
        segment_samples = []
        async for samples, _ in stream:
            total_chunks += 1
            segment_samples.append(samples)
            
        if segment_samples:
            concatenated_segment = np.concatenate(segment_samples)
            all_samples_list.append(concatenated_segment)
        
    if not all_samples_list:
        print("Error: No audio data received from any segment.")
        sys.exit(1)

    # --- Concatenate All Segments and Convert Audio Data ---
    print("\nConcatenating all audio segments...")
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
    parser = argparse.ArgumentParser(description="Generate speech from a text file using Kokoro TTS and save as M4A. Supports segmenting text by voice tags.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument(
        "voice",
        nargs='?',
        default=DEFAULT_VOICE,
        help=f"Voice name to use (default: {DEFAULT_VOICE}). This is the voice used if the input file does not start with a voice tag."
    )

    args = parser.parse_args()

    # --- Validate Input File ---
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # --- Validate Voice (Only validates the fallback voice) ---
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
    output_path = f"{base_name}_segmented.m4a" # CHANGED OUTPUT NAME
    
    # --- Run Main Function ---
    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_path}")
    asyncio.run(main(text_content, args.voice, output_path))

    print("Script finished.")