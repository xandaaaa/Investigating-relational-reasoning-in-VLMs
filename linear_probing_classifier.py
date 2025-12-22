import numpy as np
import pandas as pd
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
EVAL_CSV = Path("eval_results/evaluation_results.csv")

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


# =========================
# Load evaluation results
# =========================
inference_dataset = pd.read_csv(EVAL_CSV)
# Extract image_id from filename
inference_dataset['image_id'] = inference_dataset['image_filename'].apply(
    lambda x: int(x.split('_')[1].split('.')[0])
)
# Create is_correct column
inference_dataset['is_correct'] = np.where(
    inference_dataset['evaluation'] == "correct", 1, 0
)

inference_dataset['question_index'] = inference_dataset.groupby('image_filename').cumcount()

# Create a lookup dictionary for correctness
# Key: (image_id, question_index)
correctness_lookup = {}
for _, row in inference_dataset.iterrows():
    key = (row['image_id'], row['question_index'])
    correctness_lookup[key] = row['is_correct']


# =========================
# Load dataset JSON
# =========================
with open(QUERIES_JSON, "r") as f:
    dataset = json.load(f)

# For simple query types: features, labels, and correctness flags
X_simple = defaultdict(list)
y_simple = defaultdict(list)
correctness_simple = defaultdict(list)

# For complex query types: features, labels, and correctness flags
X_complex = defaultdict(list)
y_complex = defaultdict(lambda: defaultdict(list))
correctness_complex = defaultdict(list)


# =========================
# Build datasets grouped by query_type WITH correctness information
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

        # Get correctness for this sample
        correctness_key = (image_id, qi)
        if correctness_key not in correctness_lookup:
            continue
        is_correct = correctness_lookup[correctness_key]

        npz = np.load(npz_path)
        num_layers = int(npz.get("num_layers", 0))

        layer_features = []
        for i in range(num_layers):
            key = f"layer_{i}_mean_pooled_heads"
            if key in npz:
                layer_features.append(npz[key].astype(np.float32).flatten())

        if layer_features:
            feat = np.concatenate(layer_features)
        else:
            continue
        
        query_type = q["query_type"]
        
        # Build semantic label(s)
        options = parse_options_from_query(q["query"])
        gt_letter = q["ground_truth"]
        answer_text = options[gt_letter]
        semantic_labels = build_semantic_label(q, answer_text)
        
        # Check if structured (dict) or simple (string)
        if isinstance(semantic_labels, dict):
            # Complex type with multiple sub-labels
            X_complex[query_type].append(feat)
            correctness_complex[query_type].append(is_correct)
            for label_name, label_value in semantic_labels.items():
                y_complex[query_type][label_name].append(label_value)
        else:
            # Simple type with single label
            X_simple[query_type].append(feat)
            y_simple[query_type].append(semantic_labels)
            correctness_simple[query_type].append(is_correct)


# =========================
# Function to run probing with inference type separation
# =========================
def probe_with_inference_split(X_q, y_enc, correctness_arr, query_type, label_name=None):
    """
    Probe and report accuracy for:
    - All inference
    - Right inference (correct predictions)
    - False inference (incorrect predictions)
    """
    TEST_SIZE = 0.2
    
    unique, counts = np.unique(y_enc, return_counts=True)
    keep_classes = unique[counts >= 2]

    if len(keep_classes) < 2:
        return None

    mask = np.isin(y_enc, keep_classes)
    X_q = X_q[mask]
    y_enc = y_enc[mask]
    correctness_arr = correctness_arr[mask]

    N = len(y_enc)
    n_classes = len(keep_classes)
    
    test_size_count = int(np.floor(TEST_SIZE * N))
    if test_size_count < n_classes:
        return None

    X_train, X_test, y_train, y_test, _, correct_test = train_test_split(
        X_q, y_enc, correctness_arr, test_size=TEST_SIZE, random_state=0, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(C=0.001, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    # Calculate accuracies
    acc_all = accuracy_score(y_test, y_pred)
    
    # Right inference (where model predicted correctly)
    right_mask = correct_test == 1
    if right_mask.sum() > 0:
        acc_right = accuracy_score(y_test[right_mask], y_pred[right_mask])
        n_right = right_mask.sum()
    else:
        acc_right = None
        n_right = 0
    
    # False inference (where model predicted incorrectly)
    false_mask = correct_test == 0
    if false_mask.sum() > 0:
        acc_false = accuracy_score(y_test[false_mask], y_pred[false_mask])
        n_false = false_mask.sum()
    else:
        acc_false = None
        n_false = 0
    
    return {
        'query_type': query_type,
        'label_name': label_name,
        'n_classes': n_classes,
        'n_samples': N,
        'n_test': len(y_test),
        'n_right': n_right,
        'n_false': n_false,
        'acc_all': acc_all,
        'acc_right': acc_right,
        'acc_false': acc_false
    }


# =========================
# Linear probing for SIMPLE query types with inference split
# =========================
print("="*80)
print("SIMPLE QUERY TYPES (single label) - WITH INFERENCE TYPE SEPARATION")
print("="*80)

results_simple = []

for query_type, X_list in X_simple.items():
    X_q = np.stack(X_list, axis=0)
    y_text = np.array(y_simple[query_type])
    correctness_arr = np.array(correctness_simple[query_type])

    print(f"\n=== {query_type} ===")
    print(f"Total samples: {len(y_text)}")
    print(f"  Correct: {correctness_arr.sum()} | Incorrect: {(1-correctness_arr).sum()}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_text)

    result = probe_with_inference_split(X_q, y_enc, correctness_arr, query_type)
    
    if result:
        results_simple.append(result)
        print(f"\nAccuracy (All):   {result['acc_all']:.4f} (n={result['n_test']})")
        if result['acc_right'] is not None:
            print(f"Accuracy (Right): {result['acc_right']:.4f} (n={result['n_right']})")
        else:
            print(f"Accuracy (Right): N/A (n=0)")
        if result['acc_false'] is not None:
            print(f"Accuracy (False): {result['acc_false']:.4f} (n={result['n_false']})")
        else:
            print(f"Accuracy (False): N/A (n=0)")
    else:
        print("Skipped: insufficient data")


# =========================
# Linear probing for COMPLEX query types with inference split
# =========================
print("\n" + "="*80)
print("COMPLEX QUERY TYPES (structured multi-label) - WITH INFERENCE TYPE SEPARATION")
print("="*80)

results_complex = []

for query_type, X_list in X_complex.items():
    X_q = np.stack(X_list, axis=0)
    correctness_arr = np.array(correctness_complex[query_type])
    
    print(f"\n=== {query_type} ===")
    print(f"Total samples: {len(X_q)}")
    print(f"  Correct: {correctness_arr.sum()} | Incorrect: {(1-correctness_arr).sum()}")
    
    # Probe each sub-label separately
    for label_name, y_list in y_complex[query_type].items():
        y_text = np.array(y_list)
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y_text)
        
        result = probe_with_inference_split(X_q, y_enc, correctness_arr, query_type, label_name)
        
        if result:
            results_complex.append(result)
            print(f"\n  {label_name}:")
            print(f"    Accuracy (All):   {result['acc_all']:.4f} (n={result['n_test']})")
            if result['acc_right'] is not None:
                print(f"    Accuracy (Right): {result['acc_right']:.4f} (n={result['n_right']})")
            else:
                print(f"    Accuracy (Right): N/A (n=0)")
            if result['acc_false'] is not None:
                print(f"    Accuracy (False): {result['acc_false']:.4f} (n={result['n_false']})")
            else:
                print(f"    Accuracy (False): N/A (n=0)")


# =========================
# Create summary DataFrames and save to CSV
# =========================
print("\n" + "="*80)
print("SUMMARY - Saving results to CSV")
print("="*80)

# Simple results
df_simple = pd.DataFrame(results_simple)
df_simple.to_csv("eval_results/probing_simple_with_inference_split.csv", index=False)
print("\nSimple query results saved to: eval_results/probing_simple_with_inference_split.csv")
print(df_simple[['query_type', 'acc_all', 'acc_right', 'acc_false', 'n_right', 'n_false']])

# Complex results
df_complex = pd.DataFrame(results_complex)
df_complex.to_csv("eval_results/probing_complex_with_inference_split.csv", index=False)
print("\nComplex query results saved to: eval_results/probing_complex_with_inference_split.csv")
print(df_complex[['query_type', 'label_name', 'acc_all', 'acc_right', 'acc_false', 'n_right', 'n_false']])

print("\n" + "="*80)
print("DONE!")
print("="*80)