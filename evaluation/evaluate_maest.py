import pandas as pd
import pickle

PRED_CSV_PATH = "evaluation/maest_top_3_predictions.csv"
GT_PICKLE_PATH = "evaluation/groundtruth.pk"

predictions_df = pd.read_csv(PRED_CSV_PATH)
with open(GT_PICKLE_PATH, "rb") as f:
    ground_truth = pickle.load(f)

FILENAME_COL = "file"

def subgenre(full_string):
    parts = str(full_string).split("---", maxsplit=1)
    return parts[-1] if len(parts) == 2 else full_string

def normalize_genre(name):
    return str(name).lower().strip()


###########################################################################
#Top-1 and Top-3
###########################################################################

# global metrics for accuracy
total_tracks = len(predictions_df)
top1_correct = 0
top3_correct = 0

for idx, row in predictions_df.iterrows():
    fname = row[FILENAME_COL]
    gt_entry = ground_truth.get(fname)
    if gt_entry is None:
        continue

    gt_genre = gt_entry[0] if isinstance(gt_entry, (tuple, list)) else gt_entry
    gt_genre = normalize_genre(gt_genre)

    top1_sub = normalize_genre(subgenre(row["top1_genre"]))
    if top1_sub == gt_genre:
        top1_correct += 1

    top3_subs = [normalize_genre(subgenre(row[f"top{i}_genre"])) for i in range(1, 4)]
    if gt_genre in top3_subs:
        top3_correct += 1

# per-genre metrics for accuracy
unique_genres = sorted({
    normalize_genre(gt_entry[0] if isinstance(gt_entry, (tuple, list)) else gt_entry)
    for gt_entry in ground_truth.values()
    if gt_entry
})

per_genre_counts = {
    genre: {
        "total": 0,
        "top1_correct": 0,
        "top3_correct": 0
    }
    for genre in unique_genres
}

# count totals
for fname, gt_entry in ground_truth.items():
    gt_genre = gt_entry[0] if isinstance(gt_entry, (tuple, list)) else gt_entry
    norm_genre = normalize_genre(gt_genre)
    if norm_genre in per_genre_counts:
        per_genre_counts[norm_genre]["total"] += 1

# count correct predictions
for idx, row in predictions_df.iterrows():
    fname = row[FILENAME_COL]
    gt_entry = ground_truth.get(fname)
    if gt_entry is None:
        continue

    gt_genre = gt_entry[0] if isinstance(gt_entry, (tuple, list)) else gt_entry
    norm_genre = normalize_genre(gt_genre)

    if norm_genre not in per_genre_counts:
        continue

    top1_sub = normalize_genre(subgenre(row["top1_genre"]))
    if top1_sub == norm_genre:
        per_genre_counts[norm_genre]["top1_correct"] += 1

    top3_subs = [normalize_genre(subgenre(row[f"top{i}_genre"])) for i in range(1, 4)]
    if norm_genre in top3_subs:
        per_genre_counts[norm_genre]["top3_correct"] += 1

# global
print("Total tracks evaluated:", total_tracks)
print()

print("Top-1 Accuracy:")
print(f"  • Correct matches: {top1_correct}")
print(f"  • Accuracy: {top1_correct}/{total_tracks} = {top1_correct/total_tracks:.3f}")
print()

print("Top-3 Accuracy:")
print(f"  • Correct matches (GT in top-3): {top3_correct}")
print(f"  • Accuracy: {top3_correct}/{total_tracks} = {top3_correct/total_tracks:.3f}")
print()

# per-genre
print("Top-1 Per-Genre Accuracy:")
print("Genre".ljust(20), "Total".rjust(7), "Correct".rjust(10), "%".rjust(10))
print("-" * 50)
for genre in unique_genres:
    total = per_genre_counts[genre]["total"]
    correct = per_genre_counts[genre]["top1_correct"]
    pct = (correct / total * 100) if total > 0 else 0.0
    print(f"{genre.ljust(20)} {str(total).rjust(7)} {str(correct).rjust(10)} {pct:9.1f}")
print()

print("Top-3 Per-Genre Accuracy:")
print("Genre".ljust(20), "Total".rjust(7), "Correct".rjust(10), "%".rjust(10))
print("-" * 50)
for genre in unique_genres:
    total = per_genre_counts[genre]["total"]
    correct = per_genre_counts[genre]["top3_correct"]
    pct = (correct / total * 100) if total > 0 else 0.0
    print(f"{genre.ljust(20)} {str(total).rjust(7)} {str(correct).rjust(10)} {pct:9.1f}")


###########################################################################
#Threshold-Based
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS = "evaluation/evaluate_maest_plots"
FULL_PREDICTION_CSV = "evaluation/maest_full_predictions.csv"
full_df = pd.read_csv(FULL_PREDICTION_CSV)
genre_columns = [col for col in full_df.columns if col != "file"]

thresholds = np.arange(0.0, 1.01, 0.01)
per_genre_threshold_results = {}

for genre in genre_columns:
    genre_clean = normalize_genre(subgenre(genre))
    if genre_clean not in per_genre_counts:
        continue

    gt_positives = {
        fname for fname, gt in ground_truth.items()
        if normalize_genre(gt[0]) == genre_clean
    }

    precision_list = []
    recall_list = []
    f1_list = []

    for t in thresholds:
        tp, fp, fn = 0, 0, 0
        for idx, row in full_df.iterrows():
            fname = row["file"]
            score = row[genre]

            predicted_positive = score >= t
            is_gt = fname in gt_positives

            if predicted_positive and is_gt:
                tp += 1
            elif predicted_positive and not is_gt:
                fp += 1
            elif not predicted_positive and is_gt:
                fn += 1

        # compute values
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # optimal threshold
    best_idx = int(np.argmax(f1_list))
    best_threshold = thresholds[best_idx]
    best_f1 = f1_list[best_idx]

    per_genre_threshold_results[genre_clean] = {
        "best_threshold": best_threshold,
        "f1": best_f1,
        "precision": precision_list[best_idx],
        "recall": recall_list[best_idx]
    }

    # plot
    plt.figure()
    plt.plot(thresholds, precision_list, label="Precision")
    plt.plot(thresholds, recall_list, label="Recall")
    plt.plot(thresholds, f1_list, label="F1-score")
    plt.axvline(best_threshold, color='gray', linestyle='--', label=f'Optimal Threshold = {best_threshold:.2f}')
    plt.title(f"{genre_clean}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{genre_clean}.png"))

# results
print("\nThreshold-Based Evaluation:")
print("Genre".ljust(20), "Best Thresh".rjust(12), "F1".rjust(6), "Prec".rjust(6), "Rec".rjust(6))
print("-" * 54)
for genre, values in per_genre_threshold_results.items():
    print(f"{genre.ljust(20)} {values['best_threshold']:>12.2f} {values['f1']:>6.2f} {values['precision']:>6.2f} {values['recall']:>6.2f}")


###########################################################################
#PR-AUC
###########################################################################

from sklearn.metrics import auc

print("\nPR-AUC per Genre:")
print("Genre".ljust(20), "PR-AUC".rjust(8))
print("-" * 30)

for genre in genre_columns:
    genre_clean = normalize_genre(subgenre(genre))
    if genre_clean not in per_genre_counts:
        continue

    precision_list = []
    recall_list = []

    gt_positives = {
        fname for fname, gt in ground_truth.items()
        if normalize_genre(gt[0]) == genre_clean
    }

    for t in thresholds:
        tp, fp, fn = 0, 0, 0
        for idx, row in full_df.iterrows():
            fname = row["file"]
            score = row[genre]

            predicted_positive = score >= t
            is_gt = fname in gt_positives

            if predicted_positive and is_gt:
                tp += 1
            elif predicted_positive and not is_gt:
                fp += 1
            elif not predicted_positive and is_gt:
                fn += 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    pr_auc = auc(recall_list, precision_list)
    per_genre_threshold_results[genre_clean]['pr_auc'] = pr_auc
    print(f"{genre_clean.ljust(20)} {pr_auc:>8.3f}")

    # plot
    plt.figure()
    plt.plot(recall_list, precision_list, label=f"PR-AUC = {pr_auc:.2f}")
    plt.fill_between(recall_list, precision_list, alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{genre_clean}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{genre_clean}_pr_auc.png"))