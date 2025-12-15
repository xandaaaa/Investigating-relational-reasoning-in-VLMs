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
ROOT = Path("eval_results/attention_per_layer")
QUERIES_JSON = Path("prompts/queries.json")


# =========================
# Helper: parse options from query text
# =========================
def parse_options_from_query(query: str):
    pattern = r"([a-d]\))\s*(.*?)(?=\s+[a-d]\)|\s+Please only reply|$)"
    matches = re.findall(pattern, query)
    return {letter: text.strip() for letter, text in matches}


# =========================
# Helpers: extract concept from query text
# =========================
def extract_shape_from_query(query: str):
    m = re.search(r"have a ([a-z]+)(?: shape)?", query)
    if m:
        return m.group(1)
    return None


def extract_color_from_query(query: str):
    m = re.search(r"have a ([a-z]+) shape", query)
    if m:
        return m.group(1)
    return None


def extract_color_and_shape_from_query(query: str):
    m = re.search(r"have a ([a-z]+) ([a-z]+)", query)
    if m:
        color, shape = m.group(1), m.group(2)
        return color, shape
    return None, None


def build_semantic_label(q: dict, answer: str) -> str:
    qtype = q["query_type"]
    query_text = q["query"]
    ans_low = answer.strip().lower()

    if qtype == "recognition_shape":
        shape = extract_shape_from_query(query_text)
        if shape is None:
            return answer
        suffix = "present" if ans_low == "yes" else "absent"
        return f"{shape}_{suffix}"

    if qtype == "recognition_color":
        color = extract_color_from_query(query_text)
        if color is None:
            return answer
        suffix = "present" if ans_low == "yes" else "absent"
        return f"{color}_{suffix}"

    if qtype == "recognition_shape_and_color":
        color, shape = extract_color_and_shape_from_query(query_text)
        if color is None or shape is None:
            return answer
        combo = f"{color}_{shape}"
        suffix = "present" if ans_low == "yes" else "absent"
        return f"{combo}_{suffix}"

    if qtype == "count":
        return answer

    return answer


def extract_layer_features(npz, layer_idx):
    """
    Extract features for a specific layer from the NPZ file.
    
    Returns concatenated feature vector for the given layer.
    """
    prefix = f"layer_{layer_idx}_"
    
    # Extract all available features for this layer
    mean_pooled = npz[f"{prefix}mean_pooled_heads"].astype(np.float32)
    max_pooled = npz[f"{prefix}max_pooled_heads"].astype(np.float32)
    mean_across_steps = npz[f"{prefix}mean_across_steps"].astype(np.float32)
    last_step = npz[f"{prefix}last_step"].astype(np.float32)
    
    # Entropy features
    entropy_mean_heads = np.array([npz[f"{prefix}entropy_mean_heads"]], dtype=np.float32)
    entropy_max_heads = np.array([npz[f"{prefix}entropy_max_heads"]], dtype=np.float32)
    entropy_mean_steps = np.array([npz[f"{prefix}entropy_mean_steps"]], dtype=np.float32)
    entropy_last = np.array([npz[f"{prefix}entropy_last"]], dtype=np.float32)
    
    # Concatenate all features
    feat = np.concatenate([
        mean_pooled,
        max_pooled,
        mean_across_steps,
        last_step,
        entropy_mean_heads,
        entropy_max_heads,
        entropy_mean_steps,
        entropy_last
    ])
    
    return feat


# =========================
# Load dataset JSON
# =========================
with open(QUERIES_JSON, "r") as f:
    dataset = json.load(f)

# For each (query_type, layer), store features and labels
X_all = defaultdict(lambda: defaultdict(list))  # query_type -> layer -> list of features
y_all = defaultdict(list)  # query_type -> list of semantic labels

# Detect number of layers from first NPZ file
num_layers = None

# =========================
# Build X_all and y_all grouped by query_type and layer
# =========================
for item in dataset:
    image_id = item["image_id"]
    img_folder = ROOT / f"image_{image_id:05d}"

    if not img_folder.exists():
        print(f"Missing folder: {img_folder}")
        continue

    questions = item["questions"]

    for qi, q in enumerate(questions):
        npz_path = img_folder / f"q{qi}_attention_per_layer.npz"
        if not npz_path.exists():
            continue

        npz = np.load(npz_path)
        query_type = q["query_type"]
        
        # Detect number of layers on first file
        if num_layers is None:
            layer_keys = [k for k in npz.keys() if k.startswith("layer_") and k.endswith("_mean_pooled_heads")]
            num_layers = len(layer_keys)
            print(f"Detected {num_layers} layers in the data")
        
        # Extract features for each layer
        for layer_idx in range(num_layers):
            try:
                feat = extract_layer_features(npz, layer_idx)
                X_all[query_type][layer_idx].append(feat)
            except KeyError as e:
                print(f"Warning: Missing key {e} in {npz_path}")
                continue
        
        # Build semantic label (same for all layers, only add once per question)
        options = parse_options_from_query(q["query"])
        gt_letter = q["ground_truth"]
        answer_text = options[gt_letter]
        semantic_label = build_semantic_label(q, answer_text)
        y_all[query_type].append(semantic_label)


# =========================
# Linear probing per query type AND per layer
# =========================
results = {}
TEST_SIZE = 0.2

for query_type in X_all.keys():
    results[query_type] = {}
    y_text = np.array(y_all[query_type])
    
    print(f"\n{'='*60}")
    print(f"=== Probing query type: {query_type} ===")
    print(f"{'='*60}")
    print("Total samples:", len(y_text))
    
    # Encode labels once for this query type
    le = LabelEncoder()
    y_enc = le.fit_transform(y_text)
    
    # Filter classes with < 2 samples
    unique, counts = np.unique(y_enc, return_counts=True)
    keep_classes = unique[counts >= 2]
    
    if len(keep_classes) < 2:
        print(f"Skipping query_type={query_type}: fewer than 2 classes with ≥2 samples.")
        continue
    
    mask = np.isin(y_enc, keep_classes)
    y_enc_filtered = y_enc[mask]
    n_classes = len(keep_classes)
    N = len(y_enc_filtered)
    
    test_size_count = int(np.floor(TEST_SIZE * N))
    if test_size_count < n_classes:
        print(f"Skipping query_type={query_type}: test set too small.")
        continue
    
    print(f"Samples after filtering: {N}, Classes: {n_classes}")
    
    # Now probe each layer
    for layer_idx in sorted(X_all[query_type].keys()):
        X_layer = np.stack(X_all[query_type][layer_idx], axis=0)
        X_layer = X_layer[mask]  # Apply same mask as labels
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_layer, y_enc_filtered, test_size=TEST_SIZE, random_state=0, stratify=y_enc_filtered
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=0)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[query_type][layer_idx] = acc
        
        print(f"  Layer {layer_idx:2d}: Accuracy = {acc:.4f}")


# =========================
# Print Summary
# =========================
print("\n" + "="*60)
print("FINAL RESULTS: Accuracies per query type per layer")
print("="*60)

for query_type in sorted(results.keys()):
    print(f"\n{query_type}:")
    layer_accs = [(layer_idx, acc) for layer_idx, acc in sorted(results[query_type].items())]
    for layer_idx, acc in layer_accs:
        print(f"  Layer {layer_idx:2d}: {acc:.4f}")
    
    # Find best layer
    if layer_accs:
        best_layer, best_acc = max(layer_accs, key=lambda x: x[1])
        print(f"  --> Best: Layer {best_layer} ({best_acc:.4f})")