import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# Paths
# =========================
ROOT = Path("eval_results/attention_compact")
QUERIES_JSON = Path("prompts/queries.json")


# =========================
# Helper: parse options from query text
# =========================
def parse_options_from_query(query: str):
    """
    Extract options from text like:
      'Here are your options: a) 0 b) 4 c) 3 d) 1 ...'
    or
      'Here are your options: a) Yes b) No ...'

    Returns:
        dict: {'a)': '0', 'b)': '4', 'c)': '3', 'd)': '1'}
              or {'a)': 'Yes', 'b)': 'No'}
              or full phrases for spatial / arrow options.
    """
    pattern = r"([a-d]\))\s*(.*?)(?=\s+[a-d]\)|\s+Please only reply|$)"
    matches = re.findall(pattern, query)
    return {letter: text.strip() for letter, text in matches}


# =========================
# Helpers: extract concept (shape/color/shape+color) from query text
# =========================
def extract_shape_from_query(query: str):
    # Examples:
    # "Does this image have a rectangle shape?"
    # "Does this image have a square?"
    m = re.search(r"have a ([a-z]+)(?: shape)?", query)
    if m:
        return m.group(1)
    return None


def extract_color_from_query(query: str):
    # Example:
    # "Does this image have a yellow shape?"
    m = re.search(r"have a ([a-z]+) shape", query)
    if m:
        return m.group(1)
    return None


def extract_color_and_shape_from_query(query: str):
    # Examples:
    # "Does this image have a yellow rectangle?"
    # "Does this image have a red circle?"
    m = re.search(r"have a ([a-z]+) ([a-z]+)", query)
    if m:
        color, shape = m.group(1), m.group(2)
        return color, shape
    return None, None


def build_semantic_label(q: dict, answer: str) -> str:
    """
    Build a semantic label for probing.

    q: question dict with 'query_type' and 'query'
    answer: the selected option text (e.g. "Yes", "No", "3", "above the red circle")
    """
    qtype = q["query_type"]
    query_text = q["query"]
    ans_low = answer.strip().lower()

    # ----- recognition of single shape -----
    if qtype == "recognition_shape":
        shape = extract_shape_from_query(query_text)
        if shape is None:
            return answer  # fallback
        suffix = "present" if ans_low == "yes" else "absent"
        return f"{shape}_{suffix}"

    # ----- recognition of single color -----
    if qtype == "recognition_color":
        color = extract_color_from_query(query_text)
        if color is None:
            return answer
        suffix = "present" if ans_low == "yes" else "absent"
        return f"{color}_{suffix}"

    # ----- recognition of color+shape combination -----
    if qtype == "recognition_shape_and_color":
        color, shape = extract_color_and_shape_from_query(query_text)
        if color is None or shape is None:
            return answer
        combo = f"{color}_{shape}"
        suffix = "present" if ans_low == "yes" else "absent"
        return f"{combo}_{suffix}"

    # ----- counting -----
    if qtype == "count":
        # Here the option text is already "0", "1", "2", ...
        return answer

    # ----- other types (spatial, connection, arrow, ...) -----
    # We keep the full option text, e.g. "above the red circle"
    # or "from the magenta triangle to the yellow circle".
    return answer


# =========================
# Load dataset JSON
# =========================
with open(QUERIES_JSON, "r") as f:
    dataset = json.load(f)

# For each query type, store features and labels
X_all = defaultdict(list)  # query_type -> list of feature vectors
y_all = defaultdict(list)  # query_type -> list of semantic labels (strings)


# =========================
# Build X_all and y_all grouped by query_type
# =========================
for item in dataset:
    image_id = item["image_id"]
    img_folder = ROOT / f"image_{image_id:05d}"

    if not img_folder.exists():
        print(f"Missing folder: {img_folder}")
        continue

    questions = item["questions"]

    for qi, q in enumerate(questions):
        npz_path = img_folder / f"q{qi}_attention.npz"
        if not npz_path.exists():
            # Some query indices might not have attention saved
            print(f"Skipping image_{image_id:05d} q{qi}: {npz_path.name} not found")
            continue

        npz = np.load(npz_path)

        mean_pooled = npz["mean_pooled"].astype(np.float32)       # (64,)
        max_pooled = npz["max_pooled"].astype(np.float32)         # (64,)
        last_step_pooled = npz["last_step_pooled"].astype(np.float32)  # (64,)

        ent = np.array(
            [npz["entropy_mean"], npz["entropy_max"], npz["entropy_last"]],
            dtype=np.float32
        )  # (3,)
        vtr = npz["vision_token_range"].astype(np.float32)         # (2,)

        feat = np.concatenate([mean_pooled, max_pooled, last_step_pooled, ent, vtr])
        # feat shape: (64 + 64 + 64 + 3 + 2) = (197,)
        
        # Get query type
        query_type = q["query_type"]
        X_all[query_type].append(feat)

        # Build semantic label
        options = parse_options_from_query(q["query"])
        gt_letter = q["ground_truth"]
        answer_text = options[gt_letter]          # "Yes"/"No"/"3"/spatial phrase/etc.
        semantic_label = build_semantic_label(q, answer_text)
        y_all[query_type].append(semantic_label)


# =========================
# Linear probing per query type
# =========================
results = {}

TEST_SIZE = 0.2

for query_type, X_list in X_all.items():
    X_q = np.stack(X_list, axis=0)          # (N, 197)
    y_text = np.array(y_all[query_type])   # (N,)

    print(f"\n=== Probing query type: {query_type} ===")
    print("Samples:", len(y_text))
    print("First 5 labels:", y_text[:5])

    # Encode text labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y_text)

    # Remove classes with < 2 samples (needed for stratified split)
    unique, counts = np.unique(y_enc, return_counts=True)
    keep_classes = unique[counts >= 2]

    if len(keep_classes) < 2:
        print(f"Skipping query_type={query_type}: fewer than 2 classes have at least 2 samples.")
        continue

    mask = np.isin(y_enc, keep_classes)
    X_q = X_q[mask]
    y_enc = y_enc[mask]

    N = len(y_enc)
    n_classes = len(keep_classes)
    print("Samples after filtering:", N)
    print("Number of classes:", n_classes)

    # ---- Skip if test set would be too small (test_size * N < n_classes) ----
    test_size_count = int(np.floor(TEST_SIZE * N))
    if test_size_count < n_classes:
        print(
            f"Skipping query_type={query_type}: test_size={test_size_count} is smaller than "
            f"number of classes={n_classes}."
        )
        continue

    # Split train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_q, y_enc, test_size=TEST_SIZE, random_state=0, stratify=y_enc
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train linear classifier (linear probe)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[query_type] = acc

    kept_labels_readable = [le.inverse_transform([c])[0] for c in keep_classes]

    print(f"Accuracy for query type '{query_type}': {acc:.4f}")
    print("Classes used in this probe:", kept_labels_readable)

print("\n" + "="*60)
print("FINAL RESULTS: Accuracies per query type")
print("="*60)
for query_type, acc in sorted(results.items()):
    print(f"{query_type:30s}: {acc:.4f}")