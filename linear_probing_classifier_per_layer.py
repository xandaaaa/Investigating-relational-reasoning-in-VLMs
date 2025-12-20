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

# check the npz files 
print("Checking a sample NPZ file structure...")
sample_npz_path = ROOT / "image_00000" / "q0_attention_per_layer.npz"
if sample_npz_path.exists():
    sample_npz = np.load(sample_npz_path)
    print("Keys in the sample NPZ file:")
    print(sample_npz.files)
    sample_npz.close()
else:
    print(f"Sample NPZ file not found at {sample_npz_path}")

# =========================
# Helper: parse options from query text
# =========================
def parse_options_from_query(query: str):
    pattern = r"([a-d]\))\s*(.*?)(?=\s+[a-d]\)|\s+Please only reply|$)"
    matches = re.findall(pattern, query)
    return {letter: text.strip() for letter, text in matches}


# =========================
# Extract structured information from complex labels
# =========================
def extract_shapes_and_colors(text: str):
    """
    Extract all color-shape pairs from text like:
      'the blue triangle with the yellow rectangle'
    -> [('blue', 'triangle'), ('yellow', 'rectangle')]
    """
    pattern = r"the ([a-z]+) ([a-z]+)"
    matches = re.findall(pattern, text)
    return matches


def create_multi_labels_for_connection(answer: str):
    """
    For explicit_connection, create multiple binary labels:
    - One for each unique shape type involved
    - One for each unique color involved
    - One for the number of connections
    
    Returns dict of labels.
    """
    pairs = extract_shapes_and_colors(answer)
    
    shapes = set()
    colors = set()
    for color, shape in pairs:
        shapes.add(shape)
        colors.add(color)
    
    labels = {}
    # Number of connections (binned)
    n_connections = len(answer.split(" and "))
    if n_connections == 1:
        labels['connection_count'] = 'single'
    elif n_connections == 2:
        labels['connection_count'] = 'double'
    elif n_connections <= 4:
        labels['connection_count'] = 'few'
    else:
        labels['connection_count'] = 'many'
    
    # Shapes involved
    for shape in ['triangle', 'rectangle', 'square', 'circle']:
        labels[f'has_{shape}'] = 'yes' if shape in shapes else 'no'
    
    # Colors involved
    for color in ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'purple']:
        labels[f'has_{color}'] = 'yes' if color in colors else 'no'
    
    return labels


def create_multi_labels_for_arrow(answer: str):
    """
    For explicit_arrow_direction, extract source and target:
    'from the blue triangle to the yellow rectangle'
    -> source: (blue, triangle), target: (yellow, rectangle)
    """
    match = re.search(r"from the ([a-z]+) ([a-z]+) to the ([a-z]+) ([a-z]+)", answer)
    
    labels = {}
    if match:
        source_color, source_shape, target_color, target_shape = match.groups()
        
        # Source information
        labels['source_shape'] = source_shape
        labels['source_color'] = source_color
        labels['source_combo'] = f"{source_color}_{source_shape}"
        
        # Target information
        labels['target_shape'] = target_shape
        labels['target_color'] = target_color
        labels['target_combo'] = f"{target_color}_{target_shape}"
        
        # Direction type
        if source_shape == target_shape:
            labels['same_shape'] = 'yes'
        else:
            labels['same_shape'] = 'no'
            
        if source_color == target_color:
            labels['same_color'] = 'yes'
        else:
            labels['same_color'] = 'no'
    
    return labels


def create_multi_labels_for_spatial(answer: str):
    """
    For implicit_connection (spatial relationships), extract direction and reference:
    'above the blue circle' -> direction: above, ref_color: blue, ref_shape: circle
    'below the red triangle' -> direction: below, ref_color: red, ref_shape: triangle
    """
    # Pattern: (above|below|left of|right of) the (color) (shape)
    match = re.search(r"(above|below|left of|right of) the ([a-z]+) ([a-z]+)", answer)
    
    labels = {}
    if match:
        direction, ref_color, ref_shape = match.groups()
        
        # Spatial direction
        labels['direction'] = direction.replace(' ', '_')  # 'left of' -> 'left_of'
        
        # Reference object
        labels['ref_shape'] = ref_shape
        labels['ref_color'] = ref_color
        labels['ref_combo'] = f"{ref_color}_{ref_shape}"
        
        # Directional categories
        if direction in ['above', 'below']:
            labels['axis'] = 'vertical'
        else:  # 'left of', 'right of'
            labels['axis'] = 'horizontal'
    
    return labels


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


def build_semantic_label(q: dict, answer: str):
    """
    Build semantic label(s) for probing.
    
    For simple query types: returns a string
    For complex query types (connection/arrow/spatial): returns a dict of multiple labels
    """
    qtype = q["query_type"]
    query_text = q["query"]
    ans_low = answer.strip().lower()

    # ----- recognition of single shape -----
    if qtype == "recognition_shape":
        shape = extract_shape_from_query(query_text)
        suffix = "present" if ans_low == "yes" else "absent"
        if shape is None:
            return answer
        # Return structured labels
        return {
            'answer': suffix,  # Binary: present/absent
            'concept': shape   # Which shape was queried
        }

    # ----- recognition of single color -----
    if qtype == "recognition_color":
        color = extract_color_from_query(query_text)
        suffix = "present" if ans_low == "yes" else "absent"
        if color is None:
            return answer
        # Return structured labels
        return {
            'answer': suffix,  # Binary: present/absent
            'concept': color   # Which color was queried
        }

    # ----- recognition of color+shape combination -----
    if qtype == "recognition_shape_and_color":
        color, shape = extract_color_and_shape_from_query(query_text)
        suffix = "present" if ans_low == "yes" else "absent"
        if color is None or shape is None:
            return answer
        # Return structured labels
        return {
            'answer': suffix,           # Binary: present/absent
            'queried_color': color,     # Which color was queried
            'queried_shape': shape,     # Which shape was queried
            'concept': f"{color}_{shape}"  # Combined concept
        }

    # ----- counting -----
    if qtype == "count":
        return answer

    # ----- explicit connection (STRUCTURED) -----
    if qtype == "explicit_connection":
        return create_multi_labels_for_connection(answer)

    # ----- explicit arrow direction (STRUCTURED) -----
    if qtype == "explicit_arrow_direction":
        return create_multi_labels_for_arrow(answer)

    # ----- implicit connection / spatial relationships (STRUCTURED) -----
    if qtype in ["implicit_connection", "implicit_spatial"]:
        return create_multi_labels_for_spatial(answer)

    # ----- other types -----
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
        mean_pooled
        # max_pooled,
        # mean_across_steps,
        # last_step,
        # entropy_mean_heads,
        # entropy_max_heads,
        # entropy_mean_steps,
        # entropy_last
    ])
    
    return feat


# =========================
# Load dataset JSON
# =========================
with open(QUERIES_JSON, "r") as f:
    dataset = json.load(f)

# For simple query types: features per layer and single labels
X_simple = defaultdict(lambda: defaultdict(list))  # query_type -> layer -> list of features
y_simple = defaultdict(list)  # query_type -> list of labels

# For complex query types: features per layer and multiple labels
X_complex = defaultdict(lambda: defaultdict(list))  # query_type -> layer -> list of features
y_complex = defaultdict(lambda: defaultdict(list))  # query_type -> label_name -> list of labels

# Detect number of layers from first NPZ file
num_layers = None


# =========================
# Build datasets grouped by query_type and layer
# =========================
for item in dataset:
    image_id = item["image_id"]
    img_folder = ROOT / f"image_{image_id:05d}"

    if not img_folder.exists():
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
        
        # Build semantic label(s)
        options = parse_options_from_query(q["query"])
        gt_letter = q["ground_truth"]
        answer_text = options[gt_letter]
        semantic_labels = build_semantic_label(q, answer_text)
        
        # Extract features for each layer
        for layer_idx in range(num_layers):
            try:
                feat = extract_layer_features(npz, layer_idx)
                
                # Check if structured (dict) or simple (string)
                if isinstance(semantic_labels, dict):
                    # Complex type with multiple sub-labels
                    X_complex[query_type][layer_idx].append(feat)
                else:
                    # Simple type with single label
                    X_simple[query_type][layer_idx].append(feat)
                    
            except KeyError as e:
                print(f"Warning: Missing key {e} in {npz_path}")
                continue
        
        # Store labels (only once, not per layer)
        if isinstance(semantic_labels, dict):
            for label_name, label_value in semantic_labels.items():
                y_complex[query_type][label_name].append(label_value)
        else:
            y_simple[query_type].append(semantic_labels)


# =========================
# Linear probing for SIMPLE query types (per layer)
# =========================
print("="*60)
print("SIMPLE QUERY TYPES (single label per layer)")
print("="*60)

results_simple = {}
TEST_SIZE = 0.2

for query_type in X_simple.keys():
    results_simple[query_type] = {}
    y_text = np.array(y_simple[query_type])
    
    print(f"\n{'='*60}")
    print(f"=== {query_type} ===")
    print(f"{'='*60}")
    print("Total samples:", len(y_text))
    
    # Encode labels once for this query type
    le = LabelEncoder()
    y_enc = le.fit_transform(y_text)
    
    # Filter classes with < 2 samples
    unique, counts = np.unique(y_enc, return_counts=True)
    keep_classes = unique[counts >= 2]
    
    if len(keep_classes) < 2:
        print(f"Skipping: fewer than 2 classes with ≥2 samples.")
        continue
    
    mask = np.isin(y_enc, keep_classes)
    y_enc_filtered = y_enc[mask]
    n_classes = len(keep_classes)
    N = len(y_enc_filtered)
    
    test_size_count = int(np.floor(TEST_SIZE * N))
    if test_size_count < n_classes:
        print(f"Skipping: test set too small.")
        continue
    
    print(f"Samples after filtering: {N}, Classes: {n_classes}")
    
    # Probe each layer
    for layer_idx in sorted(X_simple[query_type].keys()):
        X_layer = np.stack(X_simple[query_type][layer_idx], axis=0)
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
        results_simple[query_type][layer_idx] = acc
        
        print(f"  Layer {layer_idx:2d}: Accuracy = {acc:.4f}")
    
    # Find best layer
    if results_simple[query_type]:
        best_layer = max(results_simple[query_type].items(), key=lambda x: x[1])
        print(f"  --> Best: Layer {best_layer[0]} ({best_layer[1]:.4f})")


# =========================
# Linear probing for COMPLEX query types (per layer, per sub-label)
# =========================
print("\n" + "="*60)
print("COMPLEX QUERY TYPES (structured multi-label per layer)")
print("="*60)

results_complex = defaultdict(lambda: defaultdict(dict))

for query_type in X_complex.keys():
    print(f"\n{'='*60}")
    print(f"=== {query_type} ===")
    print(f"{'='*60}")
    
    # Probe each sub-label separately
    for label_name, y_list in y_complex[query_type].items():
        y_text = np.array(y_list)
        
        print(f"\n  Sub-label: {label_name}")
        print(f"  Total samples: {len(y_text)}")
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y_text)
        
        unique, counts = np.unique(y_enc, return_counts=True)
        keep_classes = unique[counts >= 2]
        
        if len(keep_classes) < 2:
            print(f"  Skipping: fewer than 2 classes")
            continue
        
        mask = np.isin(y_enc, keep_classes)
        y_enc_filtered = y_enc[mask]
        n_classes = len(keep_classes)
        N = len(y_enc_filtered)
        
        test_size_count = int(np.floor(TEST_SIZE * N))
        if test_size_count < n_classes:
            print(f"  Skipping: test set too small")
            continue
        
        print(f"  Samples after filtering: {N}, Classes: {n_classes}")
        
        # Probe each layer
        for layer_idx in sorted(X_complex[query_type].keys()):
            X_layer = np.stack(X_complex[query_type][layer_idx], axis=0)
            X_layer = X_layer[mask]  # Apply same mask
            
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
            results_complex[query_type][label_name][layer_idx] = acc
            
            print(f"    Layer {layer_idx:2d}: {acc:.4f}")
        
        # Find best layer for this sub-label
        if results_complex[query_type][label_name]:
            best_layer = max(results_complex[query_type][label_name].items(), key=lambda x: x[1])
            print(f"    --> Best: Layer {best_layer[0]} ({best_layer[1]:.4f})")


# =========================
# Final Summary
# =========================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print("\n--- SIMPLE QUERY TYPES ---")
for query_type in sorted(results_simple.keys()):
    if results_simple[query_type]:
        best_layer, best_acc = max(results_simple[query_type].items(), key=lambda x: x[1])
        print(f"{query_type:30s}: Best = Layer {best_layer:2d} ({best_acc:.4f})")

print("\n--- COMPLEX QUERY TYPES ---")
for query_type in sorted(results_complex.keys()):
    print(f"\n{query_type}:")
    for label_name in sorted(results_complex[query_type].keys()):
        if results_complex[query_type][label_name]:
            best_layer, best_acc = max(results_complex[query_type][label_name].items(), key=lambda x: x[1])
            print(f"  {label_name:25s}: Best = Layer {best_layer:2d} ({best_acc:.4f})")