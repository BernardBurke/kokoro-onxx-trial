import argparse
import os
import re
import sys

# The Kokoro model has a hard limit of ~510 phonemes.
# A safe character count is generally around 280-300 characters per segment.
# We will use 280 as a conservative warning threshold.
MAX_SAFE_CHARS = 280
VOICE_TAG_PATTERN = r'<voice=[a-z0-9_]+>'

def load_and_split_text(filepath):
    """Loads the file and splits it into segments based on tags and blank lines."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

    # 1. Clean up and standardize line endings
    content = content.replace('\r\n', '\n')

    # 2. Split content into lines or potential paragraphs (handles multi-line text blocks)
    # The split includes tags, so we can track the voice
    segments = re.split(r'(\n{2,}|' + VOICE_TAG_PATTERN + r')', content, flags=re.IGNORECASE)

    # 3. Consolidate and clean up the segments
    cleaned_segments = []
    
    # Track the current line number for reporting errors
    current_line_num = 1
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # If it's a voice tag, treat it as a break but don't check length
        if re.match(VOICE_TAG_PATTERN, segment, re.IGNORECASE):
            cleaned_segments.append({
                'text': segment,
                'is_tag': True,
                'line_start': current_line_num,
                'length': len(segment)
            })
        else:
            # Handle text segments. Replace all internal newlines with a space
            text_to_check = re.sub(r'\s+', ' ', segment)
            
            cleaned_segments.append({
                'text': text_to_check,
                'is_tag': False,
                'line_start': current_line_num,
                'length': len(text_to_check)
            })
        
        # Update line number tracker (approximate, for user reference)
        # This is an approximation since a multi-line segment only counts as one
        # starting line, but it gives the user a useful reference point.
        current_line_num += segment.count('\n') + 1

    return cleaned_segments

def analyze_segments(segments):
    """Analyzes segments and reports those exceeding the character limit."""
    long_segments = []
    
    # Filter out tags and only check true content segments
    for segment in segments:
        if not segment['is_tag']:
            if segment['length'] > MAX_SAFE_CHARS:
                long_segments.append(segment)
    
    return long_segments

def check_for_missing_periods(segments):
    """Analyzes text segments for the absence of a period (.)."""
    no_period_segments = []
    
    # Filter out tags and only check true content segments
    for segment in segments:
        if not segment['is_tag']:
            # Check if a period exists in the text
            if '.' not in segment['text'].strip():
                no_period_segments.append(segment)
    
    return no_period_segments

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes a text file for segments that might exceed the Kokoro TTS phoneme limit (approx. 510 phonemes)."
    )
    parser.add_argument("input_file", help="Path to the input text file to check.")

    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    segments = load_and_split_text(args.input_file)
