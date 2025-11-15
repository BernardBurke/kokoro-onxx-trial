import asyncio
import numpy as np
import argparse
import os
import sys
import subprocess
from datetime import timedelta # For time formatting

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
DEFAULT_SUB_FORMAT = "srt"

# --- Subtitle Helper Functions ---

def format_time(seconds, sub_format="srt"):
    """Converts a float of seconds into the SRT/VTT time format: HH:MM:SS[.,]mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000

    separator = ',' if sub_format.lower() == 'srt' else '.'
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}"

def write_subtitle_file(word_timings, output_path, sub_format="srt"):
    """
    Generates a subtitle file (SRT or VTT) from word timings.
    Groups words into entries based on a pause threshold (e.g., 0.5s of silence).
    """
    if not word_timings:
        print("Warning: No word timings to generate subtitles.")
        return

    content_lines = []
    
    # VTT files require a header
    if sub_format.lower() == 'vtt':
        content_lines.append("WEBVTT\n")
    
    # Configuration for grouping subtitles (adjust as needed)
    PAUSE_THRESHOLD = 0.5 # Seconds of silence to trigger a new subtitle block

    # Subtitle building logic
    subtitle_index = 1
    current_entry_words = [word_timings[0]]

    for i in range(1, len(word_timings)):
        prev_word = word_timings[i-1]
        current_word = word_timings[i]
        
        # Check for a significant pause between words
        # Note: This relies on the 'end' of the previous word and 'start' of the current word
        if (current_word['start'] - prev_word['end']) >= PAUSE_THRESHOLD:
            # End the current entry and start a new one
            
            # 1. Write the previous entry
            start_time = current_entry_words[0]['start']
            end_time = current_entry_words[-1]['end']
            text = " ".join([d['word'] for d in current_entry_words])
            
            if sub_format.lower() == 'srt':
                content_lines.append(f"{subtitle_index}")
            
            # Format: START --> END
            content_lines.append(f"{format_time(start_time, sub_format)} --> {format_time(end_time, sub_format)}")
            
            # Subtitle text
            content_lines.append(text + "\n") # Double newline for SRT/VTT block separation

            # 2. Reset and start the new entry
            current_entry_words = [current_word]
            subtitle_index += 1
            
        else:
            # Continue the current entry
            current_entry_words.append(current_word)

    # 3. Write the last entry after the loop finishes
    if current_entry_words:
        start_time = current_entry_words[0]['start']
        end_time = current_entry_words[-1]['end']
        text = " ".join([d['word'] for d in current_entry_words])
        
        if sub_format.lower() == 'srt':
            content_lines.append(f"{subtitle_index}")
            
        content_lines.append(f"{format_time(start_time, sub_format)} --> {format_time(end_time, sub_format)}")
        content_lines.append(text + "\n")
        
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(content_lines))
        print(f"Subtitle file generated: {output_path} ({sub_format.upper()})")
    except Exception as e:
        print(f"Error writing subtitle file: {e}")

# --- Main TTS Function ---

async def main(input_text, voice_name, output_m4a_path, output_sub_path, sub_format):
    # --- Kokoro Initialization ---
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

    print(f"Generating speech with voice: {voice_name}")
    stream = kokoro.create_stream(
        input_text,
        voice=voice_name,
        speed=1.0,
        lang="en-us",
        include_word_alignments=True, # Critical for subtitles
    )

    # --- Collect Audio Chunks and Timestamps ---
    all_samples_list = []
    all_word_timings = [] 
    
    total_duration_samples = 0
    count = 0
    
    async for samples, alignment_data in stream:
        count += 1
        print(f"Received audio chunk {count}...")
        all_samples_list.append(samples)
        
        # Calculate the time offset for this chunk
        time_offset = total_duration_samples / SAMPLE_RATE
        
        # Process Alignment Data: Offset word timings
        for alignment in alignment_data:
            word_timing = {
                'word': alignment.word,
                # Adjust start/end times by the cumulative duration of previous chunks
                'start': time_offset + alignment.start_time,
                'end': time_offset + alignment.end_time,
            }
            all_word_timings.append(word_timing)
            
        total_duration_samples += len(samples)


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
        '-f', 's16le',
        '-ar', str(SAMPLE_RATE),
        '-ac', '1',
        '-i', 'pipe:0',
        '-c:a', 'aac',
        '-vn',
        '-y',
        output_m4a_path
    ]

    try:
        subprocess.run(
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
        
    # --- Subtitle Generation ---
    write_subtitle_file(all_word_timings, output_sub_path, sub_format)

# --- Argument Parsing and Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech from a text file, save as M4A, and generate SRT/VTT subtitles.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument(
        "voice",
        nargs='?',
        default=DEFAULT_VOICE,
        help=f"Voice name to use (default: {DEFAULT_VOICE})."
    )
    parser.add_argument(
        "--format",
        choices=["srt", "vtt"],
        default=DEFAULT_SUB_FORMAT,
        help=f"Subtitle format to generate (default: {DEFAULT_SUB_FORMAT})."
    )

    args = parser.parse_args()

    # --- Validation and File Reading (kept for context) ---
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    if args.voice not in VALID_VOICES:
        print(f"Error: Invalid voice name '{args.voice}'.")
        print("Available voices are:")
        for v in VALID_VOICES:
            print(f"- {v}")
        sys.exit(1)
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        if not text_content.strip():
             print(f"Error: Input file '{args.input_file}' is empty.")
             sys.exit(1)
    except Exception as e:
        print(f"Error reading input file '{args.input_file}': {e}")
        sys.exit(1)


    # --- Determine Output Filenames ---
    base_name, _ = os.path.splitext(args.input_file)
    output_m4a_path = f"{base_name}_{args.voice}.m4a"
    # Use the selected format for the subtitle file extension
    output_sub_path = f"{base_name}_{args.voice}.{args.format}"

    # --- Run Main Function ---
    print(f"Input file: {args.input_file}")
    print(f"Output audio file: {output_m4a_path}")
    print(f"Output subtitle file: {output_sub_path} (Format: {args.format.upper()})")
    
    # Pass the new arguments to main
    asyncio.run(main(text_content, args.voice, output_m4a_path, output_sub_path, args.format))

    print("Script finished.")
