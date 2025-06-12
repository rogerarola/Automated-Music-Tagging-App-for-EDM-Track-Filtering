import os
import csv
import time
from tqdm import tqdm

import essentia
import essentia.standard as es
from maest import get_maest

AUDIO_FOLDER = 'evaluation/dataset/wav'
OUTPUT_CSV = 'evaluation/maest_full_predictions.csv'

# mapped MAEST taxonomy
GENRES = [
    "Electronic---Breakbeat",
    "Electronic---Dance-pop",
    "Electronic---Deep House",
    "Electronic---Disco",
    "Electronic---Drum n Bass",
    "Electronic---Dubstep",
    "Electronic---Electro",
    "Electronic---Grime",
    "Electronic---Hard Techno",
    "Electronic---Hardcore",
    "Electronic---House",
    "Electronic---Minimal",
    "Electronic---Progressive House",
    "Electronic---Psy-Trance",
    "Electronic---Tech House",
    "Electronic---Techno",
    "Electronic---Trance",
    "Electronic---UK Garage"
]

# Load model
print("Loading MAEST model...")
model = get_maest(arch="discogs-maest-30s-pw-129e-519l")
model.eval()
print("Model loaded.")

# Collect audio files
audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]
print(f"Found {len(audio_files)} audio files.")

# CSV output
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file"] + GENRES)

    for i, file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        path = os.path.join(AUDIO_FOLDER, file)
        start = time.time()

        try:
            audio = es.MonoLoader(filename=path, sampleRate=16000)()
            activations, labels = model.predict_labels(audio)

            # Map each genre in fixed GENRES list to its corresponding activation
            genre_score_map = dict(zip(labels, activations))
            ordered_scores = [genre_score_map.get(genre, 0.0) for genre in GENRES]

            writer.writerow([file] + ordered_scores)
            csvfile.flush()

            print(f"[{i+1}/{len(audio_files)}] {file} processed in {time.time() - start:.2f}s")

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

print(f"\n Full predictions saved to: {OUTPUT_CSV}")