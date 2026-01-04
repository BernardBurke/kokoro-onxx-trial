import sys
import os
import re

def fix_text_file(input_path):
    # 1. Validate input
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # 2. Determine output path
    # /path/to/story.txt -> /path/to/story_fixed.txt
    folder, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_fixed{ext}"
    output_path = os.path.join(folder, output_filename)

    print(f"Reading: {input_path}")

    # 3. Read and Clean Text
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Normalize whitespace: turns newlines and multiple spaces into a single space.
    # This fixes issues where a sentence is broken across two lines in the source.
    clean_text = re.sub(r'\s+', ' ', raw_text).strip()

    # 4. Split into sentences
    # Regex breakdown:
    # (?<=[.?!]) -> Lookbehind: Check if the previous char was '.', '?', or '!'
    # \s+        -> Match one or more spaces after the punctuation
    # This splits "Hello. World." into "Hello." and "World."
    sentences = re.split(r'(?<=[.?!])\s+', clean_text)

    # Filter out empty strings just in case
    sentences = [s.strip() for s in sentences if s.strip()]

    # 5. Write to new file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write one sentence per line
        f.write('\n'.join(sentences))

    print(f"Success! Fixed file created at: {output_path}")
    print(f"Total lines/sentences: {len(sentences)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_text.py <path_to_source_file>")
    else:
        fix_text_file(sys.argv[1])
