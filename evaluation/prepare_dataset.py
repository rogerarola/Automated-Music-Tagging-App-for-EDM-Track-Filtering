import os
import subprocess
import pickle
import librosa
import numpy as np

input_folder = "evaluation/dataset"
output_folder = "evaluation/dataset/wav"
os.makedirs(output_folder, exist_ok=True)
groundtruth = {}

# RMS function
def find_loudest_segment(audio_path, target_duration=30.0):
    try:
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return 0

    hop_length = 512
    frame_duration = hop_length / sr
    frames_per_window = int(target_duration / frame_duration)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    window_energies = np.convolve(rms, np.ones(frames_per_window), mode='valid')
    
    if len(window_energies) == 0:
        return 0

    best_frame = np.argmax(window_energies)
    start_sec = best_frame * frame_duration
    return start_sec

for genre in os.listdir(input_folder):
    genre_folder = os.path.join(input_folder, genre)
    if not os.path.isdir(genre_folder):
        continue

    for file in os.listdir(genre_folder):
        if not file.endswith(".mp3"):
            continue

        input_path = os.path.join(genre_folder, file)
        output_name = f"{genre}__{os.path.splitext(file)[0]}.wav"
        output_path = os.path.join(output_folder, output_name)

        # best 30s segment
        start_time = find_loudest_segment(input_path)
        subprocess.run([
            "ffmpeg", "-ss", str(start_time), "-i", input_path,
            "-t", "30", "-ac", "1", "-ar", "16000", output_path,
            "-y", "-loglevel", "error"
        ])

        groundtruth[output_name] = (genre,)

# pickle file creation
with open("evaluation/groundtruth.pk", "wb") as f:
    pickle.dump(groundtruth, f)

print(f"\n {len(groundtruth)} files converted & groundtruth.pk created")