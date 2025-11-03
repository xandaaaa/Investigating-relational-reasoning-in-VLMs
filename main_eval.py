print("Importing transformers...", flush=True)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

print("Importing torch...", flush=True)
import torch

print("Importing huggingface_hub...", flush=True)
from huggingface_hub import login

import os
print("Setting environment variables...", flush=True)

import os, json, re, csv, argparse, random
from dataclasses import dataclass  
from typing import Dict, List, Tuple
from PIL import Image

# If not logged in via CLI, login programmatically

# ----- config : Relation Templates ----------

REL_TEMPLATES = {
    "left_of":   ["Is the {a_color} {a_shape} to the left of the {b_color} {b_shape}?"],
    "right_of":  ["Is the {a_color} {a_shape} to the right of the {b_color} {b_shape}?"],
    "above":     ["Is the {a_color} {a_shape} above the {b_color} {b_shape}?"],
    "below":     ["Is the {a_color} {a_shape} below the {b_color} {b_shape}?"],
    "in_front_of": ["Is the {a_color} {a_shape} in front of the {b_color} {b_shape}?"],
    "behind":    ["Is the {a_color} {a_shape} behind the {b_color} {b_shape}?"],
}
YES_WORDS = {"yes", "true", "correct", "yeah", "yep"}
NO_WORDS  = {"no", "false", "incorrect", "nope"}

@dataclass
class Ent:
    id: int
    shape: str
    size: str
    color: str

# ---------------- Helpers for your JSON schema ----------------
def parse_annotation(j: Dict) -> Tuple[str, Dict[int, Ent], List[Dict]]:
    """
    Your JSON keys:
      image_filename, entities[{id,shape,size,color,...}], relations[{subject_id,object_id,relation,...}]
    Returns: (image_rel_path, id->Ent, relations list with keys subject/object/relation)
    """
    img = j["image_filename"]
    id2ent: Dict[int, Ent] = {}
    for o in j["entities"]:
        id2ent[o["id"]] = Ent(id=o["id"], shape=o["shape"], size=o.get("size",""), color=o["color"])
    rels = []
    for r in j.get("relations", []):
        rels.append({"subject": r["subject_id"], "object": r["object_id"], "relation": r["relation"]})
    return img, id2ent, rels



def make_question(id2ent: Dict[int, Ent], rel: Dict) -> Tuple[str, str]:
    """Return (question, gt='yes'). Templates are phrased to be TRUE for the given triple."""
    r = rel["relation"]
    if r not in REL_TEMPLATES: 
        return None, None
    a, b = id2ent[rel["subject"]], id2ent[rel["object"]]
    tpl = random.choice(REL_TEMPLATES[r])
    q = tpl.format(a_color=a.color, a_shape=a.shape, b_color=b.color, b_shape=b.shape)
    return q, "yes"


# ---------------- VLM runners ----------------
@torch.inference_mode()
def run_vlm(model, processor, image_path: str, question: str, device="cuda") -> str:
    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question}
    ]}]
    inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    return output_text[0]


@torch.inference_mode()
def run_vlm_text_only(model, processor, question: str, device="cuda") -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    inputs = processor.apply_chat_template(msgs, add_generation_prompt=True,  tokenize = True, return_dict=True, return_tensors="pt").to(device)
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    return output_text[0]

def parse_yesno(text: str) -> str:
    t = text.lower()
    t = re.split(r'[.!?\n]', t)[0] if len(t) > 80 else t
    if any(w in t for w in YES_WORDS) and not any(w in t for w in NO_WORDS): return "yes"
    if any(w in t for w in NO_WORDS) and not any(w in t for w in YES_WORDS): return "no"
    tok0 = (t.strip().split()[:1] or [""])[0]
    if tok0 in YES_WORDS: return "yes"
    if tok0 in NO_WORDS:  return "no"
    return "unknown"



def evaluate(rows: List[Dict]) -> Dict:
    n = len(rows)
    acc = sum(r["pred"] == r["gt"] for r in rows) / max(1, n)
    by_rel: Dict[str, List[int]] = {}
    for r in rows:
        by_rel.setdefault(r["relation"], []).append(1 if r["pred"] == r["gt"] else 0)
    per_rel = {k: sum(v)/len(v) for k, v in by_rel.items()}
    return {"n": n, "acc": acc, "per_relation": per_rel}

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=False, help="Root folder of images")
    ap.add_argument("--ann_dir", required=False, help="Folder with JSON annotations")
    ap.add_argument("--out_dir", default="./rel_out")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--text_only_baseline", action="store_true")
    args = ap.parse_args()

    login(token="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")
    print("logged into HF", flush=True)


    print("Loading Qwen3-VL-4B-Instruct...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    device = model.device

        # If no dataset args are provided, keep your original single-image demo
    if not (args.img_dir and args.ann_dir):
        print("No dataset provided; running single-image demo...", flush=True)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": "foodchain.jpeg"},
                {"type": "text", "text": "Describe this image. List components and explain relations in sequence."},
            ],
        }]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_new_tokens=512)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(text)
        return

    # --------- Dataset evaluation ----------
    # collect JSONs
    json_files = []
    for root, _, files in os.walk(args.ann_dir):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    if not json_files:
        raise FileNotFoundError(f"No .json under {args.ann_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    rows, seen = [], 0

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]

        for j in items:
            img_rel, id2ent, rels = parse_annotation(j)
            if not rels: 
                continue
            img_path = img_rel if os.path.isabs(img_rel) else os.path.join(args.img_dir, img_rel)
            if not os.path.exists(img_path):
                continue

            for rel in rels:
                if rel["relation"] not in REL_TEMPLATES:
                    continue
                q, gt = make_question(id2ent, rel)
                if not q:
                    continue

                # run VLM
                print('\n\n debug: ----------------')
                print(q)
                raw = run_vlm(model, processor, img_path, q, device=str(device))
                pred = parse_yesno(raw)

                rec = {
                    "img": img_path, "question": q, "gt": gt, "pred": pred,
                    "relation": rel["relation"], "raw": raw
                }

                if args.text_only_baseline:
                    raw_txt = run_vlm_text_only(model, processor, q, device=str(device))
                    rec["pred_text_only"] = parse_yesno(raw_txt)
                    rec["raw_text_only"] = raw_txt

                rows.append(rec)
                seen += 1
                if seen >= args.max_samples:
                    break
            if seen >= args.max_samples:
                break
        if seen >= args.max_samples:
            break

    metrics = evaluate(rows)
    print("\n=== Relational Accuracy ===", flush=True)
    print(f"Samples: {metrics['n']}", flush=True)
    print(f"Overall: {metrics['acc']:.3f}", flush=True)
    for k, v in sorted(metrics["per_relation"].items()):
        print(f"  {k:12s}: {v:.3f}", flush=True)

    csv_path = os.path.join(args.out_dir, "rel_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img","question","gt","pred","relation","raw","pred_text_only","raw_text_only"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved per-sample results → {csv_path}", flush=True)

if __name__ == "__main__":
    main()            



# login(token="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")
# print("logged into HF", flush=True)

# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU name: {torch.cuda.get_device_name(0)}")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
# else:
#     print("⚠️ No GPU detected. Running on CPU.")

# # default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "foodchain.jpeg",
#             },
#             {"type": "text", "text": "Describe this image. list all the components in the image and Explain the relation in sequence"},
#         ],
#     }
# ]

# # Preparation for inference
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt"
# )
# inputs = inputs.to(model.device)

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=1024)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)
