"""
Generates synthetic labeled training data for the trained classifier
scorer. Each example is a (actual_output, expected_output) pair with
engineered features plus a synthetic 0/1 label, built from controlled
perturbations so the label is a known ground truth we can train against.

Run manually: python -m app.ml.generate_training_data
"""
import csv
import random

from app.ml.features import word_overlap_ratio, length_ratio

GOOD_PAIRS = [
    ("Paris is the capital of France.", "Paris is the capital of France."),
    ("The capital of France is Paris.", "Paris is the capital of France."),
    ("Water boils at 100 degrees Celsius.", "Water boils at 100 degrees Celsius at sea level."),
    ("The mitochondria is the powerhouse of the cell.", "Mitochondria produce energy for the cell."),
    ("Shakespeare wrote Romeo and Juliet.", "Romeo and Juliet was written by Shakespeare."),
    ("The Great Wall of China is over 13000 miles long.", "The Great Wall of China stretches over 13000 miles."),
    ("Photosynthesis converts sunlight into energy.", "Plants use photosynthesis to convert light into energy."),
    ("The Pacific Ocean is the largest ocean on Earth.", "The largest ocean on Earth is the Pacific."),
    ("Albert Einstein developed the theory of relativity.", "The theory of relativity was developed by Einstein."),
    ("A triangle has three sides.", "Triangles have three sides."),
]

BAD_PAIRS = [
    ("The sky is green.", "Paris is the capital of France."),
    ("Bananas are a type of vehicle.", "Water boils at 100 degrees Celsius at sea level."),
    ("The moon is made of cheese.", "Mitochondria produce energy for the cell."),
    ("Dogs can fly using their ears.", "Romeo and Juliet was written by Shakespeare."),
    ("The internet was invented in 1400.", "The Great Wall of China stretches over 13000 miles."),
    ("Fish live in trees.", "Plants use photosynthesis to convert light into energy."),
    ("The sun orbits the Earth.", "The largest ocean on Earth is the Pacific."),
    ("Two plus two equals five.", "The theory of relativity was developed by Einstein."),
    ("A square has seven sides.", "Triangles have three sides."),
    ("London is the capital of Japan.", "Paris is the capital of France."),
]


def generate(output_path: str = "training_data.csv", augment_factor: int = 20):
    rows = []
    random.seed(42)

    for label, pairs in [(1, GOOD_PAIRS), (0, BAD_PAIRS)]:
        for actual, expected in pairs:
            for _ in range(augment_factor):
                words = actual.split()
                if len(words) > 3 and random.random() < 0.3:
                    i, j = random.sample(range(len(words)), 2)
                    words[i], words[j] = words[j], words[i]
                jittered = " ".join(words)

                rows.append({
                    "actual_output": jittered,
                    "expected_output": expected,
                    "word_overlap": word_overlap_ratio(jittered, expected),
                    "length_ratio": length_ratio(jittered, expected),
                    "label": label,
                })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["actual_output", "expected_output", "word_overlap", "length_ratio", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    generate()