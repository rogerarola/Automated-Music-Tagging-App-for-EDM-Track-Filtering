import os
import csv
import time
from tqdm import tqdm

import essentia
import essentia.standard as es
from maest import get_maest

AUDIO_FOLDER = 'evaluation/dataset/wav'
OUTPUT_CSV = 'evaluation/maest_top_3_predictions.csv'

# load MAEST
print("Loading MAEST model...")
model = get_maest(arch="discogs-maest-30s-pw-129e-519l")
model.eval()
print("Model loaded.")

# collect wav files
audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]
print(f"Found {len(audio_files)} audio files.")

# CSV creation
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'file',
        'top1_genre', 'score1',
        'top2_genre', 'score2',
        'top3_genre', 'score3'
    ])

    for i, file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        path = os.path.join(AUDIO_FOLDER, file)
        start = time.time()

        try:
            audio = es.MonoLoader(filename=path, sampleRate=16000)()
            activations, labels = model.predict_labels(audio)

            top_indices = activations.argsort()[::-1][:3]
            top_genres  = [labels[i] for i in top_indices]
            top_scores  = [activations[i] for i in top_indices]

            writer.writerow([
                file,
                top_genres[0], top_scores[0],
                top_genres[1], top_scores[1],
                top_genres[2], top_scores[2]
            ])

            csvfile.flush()
            print(f"[{i+1}/{len(audio_files)}] {file} → {top_genres[0]} ({top_scores[0]:.3f}) in {time.time() - start:.2f}s")

        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

print(f"\nPredictions saved to {OUTPUT_CSV}")