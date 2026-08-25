"""
Converts the Criteo Kaggle Display Advertising Challenge train.txt into a
preprocessed criteo.csv suitable for the VFL framework.

Input:  train.txt  (tab-separated, 39 fields: Label + 13 int + 26 cat)
Output: criteo.csv (D_=8192 hashed features + label column, 100k rows)

Feature engineering (from https://github.com/swapniel99/criteo/blob/master/criteo.py):
  - Each of the 39 feature fields is hashed into a D_-dimensional space
  - All pairwise XOR cross-features are added, giving 39 + 741 = 780 active indices per row
  - Result: dense binary-count vector of length 8192

Sampling strategy: alternates label 0 / label 1 to produce a balanced dataset.
Total output rows: MAX_SAMPLE_NUM = 100,000.

Run from the Code/ directory:
    python datasets_preprocess/criteo_preprocess.py

Output file size: ~3 GB (dense text CSV, 8193 columns).
Estimated runtime: 15–30 minutes (CPU-bound, scanning ~45M rows of train.txt).
"""
import csv
from csv import DictReader

D_ = 2 ** 13          # hashed feature dimension = 8192
MAX_SAMPLE_NUM = 100_000

HEADER = [
    'Label',
    'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13',
    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13',
    'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26',
]


def get_x(csv_row, D):
    """Hash all feature fields + pairwise XOR cross-features into a D-dimensional count vector."""
    fullind = [hash(key + '=' + value) % D for key, value in csv_row.items()]

    # Pairwise XOR feature crossing
    n = len(fullind)
    cross = [fullind[i] ^ fullind[j] for i in range(n) for j in range(i + 1, n)]
    fullind = fullind + cross

    x = [0.] * D
    for idx in fullind:
        x[idx] += 1
    return x


if __name__ == '__main__':
    import os
    train_txt_file_path = r'.\data\kaggle-display-advertising-challenge-dataset\train.txt'
    output_dir          = r'.\data\Criteo'
    output_file_path    = os.path.join(output_dir, 'criteo.csv')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading:  {train_txt_file_path}")
    print(f"Writing:  {output_file_path}")
    print(f"Rows:     {MAX_SAMPLE_NUM:,} (alternating label 0/1 for class balance)")
    print(f"Columns:  {D_} hashed features + 1 label = {D_ + 1}")
    print(f"WARNING:  Output file will be ~3 GB. Runtime ~15-30 min.")
    print()

    reader = DictReader(open(train_txt_file_path, encoding='utf-8'), HEADER, delimiter='\t')

    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow([f'feat{i}' for i in range(D_)] + ['label'])

        count = 0
        pre_label = 1.  # start expecting label=0 first (triggers on change from 1→0)

        for row in reader:
            y = 1. if row['Label'] == '1' else 0.

            # Only write when label changes — ensures alternating 0/1 balance
            if y != pre_label:
                pre_label = y
                count += 1
                del row['Label']
                x = get_x(row, D_)
                x.append(y)
                writer.writerow(x)

                if count % 1000 == 0:
                    pct = 100 * count / MAX_SAMPLE_NUM
                    print(f"  {pct:.1f}% — {count:,}/{MAX_SAMPLE_NUM:,} rows written")

                if count == MAX_SAMPLE_NUM:
                    break

    print(f"\nDone. Wrote {count:,} rows to:\n  {output_file_path}")
