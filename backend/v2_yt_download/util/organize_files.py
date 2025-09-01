import os
import shutil
from pathlib import Path

# Define paths to the "downloads" directory and subfolders
downloads_dir = Path("../downloads")  # Replace with your actual "downloads" path
audio_dir = downloads_dir / "audio"
metadata_dir = downloads_dir / "metadata"

# Create the audio and metadata directories if they don't exist
audio_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)


# Function to move files into respective directories
def move_files():
    # Check all files in the main downloads directory (excluding subdirectories like 'audio')
    for item in downloads_dir.iterdir():
        if item.is_file():  # Only process actual files, skip directories and symlinks
            print(f"Checking file: {item} | Suffix: {item.suffix}")  # Debugging line to check files
            file_name = item.name.strip()  # Strip any leading/trailing spaces
            if file_name.endswith(".wav"):
                # Move .wav files to the audio directory
                print(f"Moving WAV file: {item.name}")  # Debugging line to confirm file type
                shutil.move(str(item), str(audio_dir / item.name))
            elif file_name.endswith(".djmeta.json") or file_name.endswith(".csv"):
                # Move .djmeta.json and .csv files to the metadata directory
                print(f"Moving metadata file: {item.name}")  # Debugging line to confirm file type
                shutil.move(str(item), str(metadata_dir / item.name))
        elif item.is_dir():  # If it's a directory, skip or check inside
            continue  # Skip the subdirectories

# Run the function to organize files
move_files()

print("Files have been organized into 'audio' and 'metadata' directories.")