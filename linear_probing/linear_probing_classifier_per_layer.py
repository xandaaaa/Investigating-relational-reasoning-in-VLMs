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
    pattern = r"the ([a-z]+) ([a-z]+)"
    matches = re.findall(pattern, text)
    return matches


def create_multi_labels_for_connection(answer: str):
    pairs = extract_shapes_and_colors(answer)
    
    shapes = set()
    colors = set()
    for color, shape in pairs:
        shapes.add(shape)
        colors.add(color)
    
    labels = {}
    n_connections = len(answer.split(" and "))
    if n_connections == 1:
        labels['connection_count'] = 'single'
    elif n_connections == 2:
        labels['connection_count'] = 'double'
    elif n_connections <= 4:
        labels['connection_count'] = 'few'
    else:
        labels['connection_count'] = 'many'
    
    for shape in ['triangle', 'rectangle', 'square', 'circle']:
        labels[f'has_{shape}'] = 'yes' if shape in shapes else 'no'
    
    for color in ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'purple']:
        labels[f'has_{color}'] = 'yes' if color in colors else 'no'
    
    return labels


def create_multi_labels_for_arrow(answer: str):
    match = re.search(r"from the ([a-z]+) ([a-z]+) to the ([a-z]+) ([a-z]+)", answer)
    
    labels = {}
    if match:
        source_color, source_shape, target_color, target_shape = match.groups()
        
        labels['source_shape'] = source_shape
        labels['source_color'] = source_color
        labels['source_combo'] = f"{source_color}_{source_shape}"
        
        labels['target_shape'] = target_shape
        labels['target_color'] = target_color
        labels['target_combo'] = f"{target_color}_{target_shape}"
        
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
    match = re.search(r"(above|below|left of|right of) the ([a-z]+) ([a-z]+)", answer)
    
    labels = {}
    if match:
        direction, ref_color, ref_shape = match.groups()
        
        labels['direction'] = direction.replace(' ', '_')
        labels['ref_shape'] = ref_shape
        labels['ref_color'] = ref_color
        labels['ref_combo'] = f"{ref_color}_{ref_shape}"
        
        if direction in ['above', 'below']:
            labels['axis'] = 'vertical'
        else:
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
    qtype = q["query_type"]
    query_text = q["query"]
    ans_low = answer.strip().lower()

    if qtype == "recognition_shape":
        shape = extract_shape_from_query(query_text)
        suffix = "present" if ans_low == "yes" else "absent"
        if shape is None:
            return answer
        return {
            'answer': suffix,
            'concept': shape
        }

    if qtype == "recognition_color":
        color = extract_color_from_query(query_text)
        suffix = "present" if ans_low == "yes" else "absent"
        if color is None:
            return answer
        return {
            'answer': suffix,
            'concept': color
        }

    if qtype == "recognition_shape_and_color":
        color, shape = extract_color_and_shape_from_query(query_text)
        suffix = "present" if ans_low == "yes" else "absent"
        if color is None or shape is None:
            return answer
        return {
            'answer': suffix,
            'queried_color': color,
            'queried_shape': shape,
            'concept': f"{color}_{shape}"
        }

    if qtype == "count":
        return answer

    if qtype == "explicit_connection":
        return create_multi_labels_for_connection(answer)

    if qtype == "explicit_arrow_direction":
        return create_multi_labels_for_arrow(answer)

    if qtype in ["implicit_connection", "implicit_spatial"]:
        return create_multi_labels_for_spatial(answer)

    return answer


def extract_layer_features(npz, layer_idx):
    prefix = f"layer_{layer_idx}_"
    
    mean_pooled = npz[f"{prefix}mean_pooled_heads"].astype(np.float32)
    
    feat = np.concatenate([mean_pooled])
    
    return feat


# =========================
# Load evaluation results
# =========================
inference_dataset = pd.read_csv(EVAL_CSV)
inference_dataset['image_id'] = inference_dataset['image_filename'].apply(
    lambda x: int(x.split('_')[1].split('.')[0])
)
inference_dataset['question_index'] = inference_dataset.groupby('image_filename').cumcount()
inference_dataset['is_correct'] = np.where(
    inference_dataset['evaluation'] == "correct", 1, 0
)

correctness_lookup = {}
for _, row in inference_dataset.iterrows():
    key = (row['image_id'], row['question_index'])
    correctness_lookup[key] = row['is_correct']


# =========================
# Load dataset JSON
# =========================
with open(QUERIES_JSON, "r") as f:
    dataset = json.load(f)

# For simple query types
X_simple = defaultdict(lambda: defaultdict(list))
y_simple = defaultdict(list)
correctness_simple = defaultdict(list)

# For complex query types
X_complex = defaultdict(lambda: defaultdict(list))
y_complex = defaultdict(lambda: defaultdict(list))
correctness_complex = defaultdict(list)

num_layers = None


# =========================
# Build datasets
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

        # Get correctness
        correctness_key = (image_id, qi)
        if correctness_key not in correctness_lookup:
            continue
        is_correct = correctness_lookup[correctness_key]

        npz = np.load(npz_path)
        query_type = q["query_type"]
        
        if num_layers is None:
            layer_keys = [k for k in npz.keys() if k.startswith("layer_") and k.endswith("_mean_pooled_heads")]
            num_layers = len(layer_keys)
            print(f"Detected {num_layers} layers in the data")
        
        options = parse_options_from_query(q["query"])
        gt_letter = q["ground_truth"]
        answer_text = options[gt_letter]
        semantic_labels = build_semantic_label(q, answer_text)
        
        # Extract features for each layer
        for layer_idx in range(num_layers):
            try:
                feat = extract_layer_features(npz, layer_idx)
                
                if isinstance(semantic_labels, dict):
                    X_complex[query_type][layer_idx].append(feat)
                else:
                    X_simple[query_type][layer_idx].append(feat)
                    
            except KeyError as e:
                print(f"Warning: Missing key {e} in {npz_path}")
                continue
        
        # Store labels and correctness (once per sample)
        if isinstance(semantic_labels, dict):
            correctness_complex[query_type].append(is_correct)
            for label_name, label_value in semantic_labels.items():
                y_complex[query_type][label_name].append(label_value)
        else:
            y_simple[query_type].append(semantic_labels)
            correctness_simple[query_type].append(is_correct)


# =========================
# Probing function with inference split
# =========================
def probe_layer_with_inference_split(X_layer, y_enc, correctness_arr, query_type, layer_idx, label_name=None):
    TEST_SIZE = 0.2
    
    unique, counts = np.unique(y_enc, return_counts=True)
    keep_classes = unique[counts >= 2]

    if len(keep_classes) < 2:
        return None

    mask = np.isin(y_enc, keep_classes)
    X_layer = X_layer[mask]
    y_enc = y_enc[mask]
    correctness_arr = correctness_arr[mask]

    N = len(y_enc)
    n_classes = len(keep_classes)
    
    test_size_count = int(np.floor(TEST_SIZE * N))
    if test_size_count < n_classes:
        return None

    X_train, X_test, y_train, y_test, _, correct_test = train_test_split(
        X_layer, y_enc, correctness_arr, test_size=TEST_SIZE, random_state=0, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    acc_all = accuracy_score(y_test, y_pred)
    
    right_mask = correct_test == 1
    if right_mask.sum() > 0:
        acc_right = accuracy_score(y_test[right_mask], y_pred[right_mask])
        n_right = right_mask.sum()
    else:
        acc_right = None
        n_right = 0
    
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
        'layer': layer_idx,
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
# Probe SIMPLE query types
# =========================
print("="*80)
print("SIMPLE QUERY TYPES - LAYER-WISE WITH INFERENCE SPLIT")
print("="*80)

results_simple = []

for query_type in X_simple.keys():
    y_text = np.array(y_simple[query_type])
    correctness_arr = np.array(correctness_simple[query_type])
    
    print(f"\n{'='*80}")
    print(f"=== {query_type} ===")
    print(f"{'='*80}")
    print(f"Total samples: {len(y_text)}")
    print(f"  Correct: {correctness_arr.sum()} | Incorrect: {(1-correctness_arr).sum()}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_text)
    
    for layer_idx in sorted(X_simple[query_type].keys()):
        X_layer = np.stack(X_simple[query_type][layer_idx], axis=0)
        
        result = probe_layer_with_inference_split(
            X_layer, y_enc, correctness_arr, query_type, layer_idx
        )
        
        if result:
            results_simple.append(result)
            print(f"\n  Layer {layer_idx:2d}:")
            print(f"    Acc (All):   {result['acc_all']:.4f} (n={result['n_test']})")
            if result['acc_right'] is not None:
                print(f"    Acc (Right): {result['acc_right']:.4f} (n={result['n_right']})")
            if result['acc_false'] is not None:
                print(f"    Acc (False): {result['acc_false']:.4f} (n={result['n_false']})")
    
    # Find best layer for each metric
    query_results = [r for r in results_simple if r['query_type'] == query_type]
    if query_results:
        best_all = max(query_results, key=lambda x: x['acc_all'])
        print(f"\n  --> Best All: Layer {best_all['layer']} ({best_all['acc_all']:.4f})")
        
        right_results = [r for r in query_results if r['acc_right'] is not None]
        if right_results:
            best_right = max(right_results, key=lambda x: x['acc_right'])
            print(f"  --> Best Right: Layer {best_right['layer']} ({best_right['acc_right']:.4f})")
        
        false_results = [r for r in query_results if r['acc_false'] is not None]
        if false_results:
            best_false = max(false_results, key=lambda x: x['acc_false'])
            print(f"  --> Best False: Layer {best_false['layer']} ({best_false['acc_false']:.4f})")


# =========================
# Probe COMPLEX query types
# =========================
print("\n" + "="*80)
print("COMPLEX QUERY TYPES - LAYER-WISE WITH INFERENCE SPLIT")
print("="*80)

results_complex = []

for query_type in X_complex.keys():
    correctness_arr = np.array(correctness_complex[query_type])
    
    print(f"\n{'='*80}")
    print(f"=== {query_type} ===")
    print(f"{'='*80}")
    print(f"Total samples: {len(correctness_arr)}")
    print(f"  Correct: {correctness_arr.sum()} | Incorrect: {(1-correctness_arr).sum()}")
    
    for label_name, y_list in y_complex[query_type].items():
        y_text = np.array(y_list)
        
        print(f"\n  Sub-label: {label_name}")
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y_text)
        
        for layer_idx in sorted(X_complex[query_type].keys()):
            X_layer = np.stack(X_complex[query_type][layer_idx], axis=0)
            
            result = probe_layer_with_inference_split(
                X_layer, y_enc, correctness_arr, query_type, layer_idx, label_name
            )
            
            if result:
                results_complex.append(result)
                print(f"    Layer {layer_idx:2d}: All={result['acc_all']:.4f}", end="")
                if result['acc_right'] is not None:
                    print(f" | Right={result['acc_right']:.4f}", end="")
                if result['acc_false'] is not None:
                    print(f" | False={result['acc_false']:.4f}", end="")
                print()
        
        # Find best layer for this sub-label
        sublabel_results = [r for r in results_complex 
                           if r['query_type'] == query_type and r['label_name'] == label_name]
        if sublabel_results:
            best_all = max(sublabel_results, key=lambda x: x['acc_all'])
            print(f"    --> Best All: Layer {best_all['layer']} ({best_all['acc_all']:.4f})")


# =========================
# Save results to CSV
# =========================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

df_simple = pd.DataFrame(results_simple)
df_simple.to_csv("eval_results/layer_probing_simple_with_inference_split.csv", index=False)
print("\nSimple results saved to: eval_results/layer_probing_simple_with_inference_split.csv")
print(df_simple[['query_type', 'layer', 'acc_all', 'acc_right', 'acc_false', 'n_right', 'n_false']].head(20))

df_complex = pd.DataFrame(results_complex)
df_complex.to_csv("eval_results/layer_probing_complex_with_inference_split.csv", index=False)
print("\nComplex results saved to: eval_results/layer_probing_complex_with_inference_split.csv")
print(df_complex[['query_type', 'label_name', 'layer', 'acc_all', 'acc_right', 'acc_false']].head(20))

print("\n" + "="*80)
print("DONE!")
print("="*80)