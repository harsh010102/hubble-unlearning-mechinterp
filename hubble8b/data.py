"""Data pipeline: YAGO decode, three-way splits, leak-free prompts, distractor pool.

Scale-invariant — identical for 1B and 8B. Ported from the validated 1B notebooks
with every fix baked in (Unseen from the TEST split; seeded-shuffle Unlearn/Keep;
sorted(set) distractor pool; leak-free prefixes).
"""
import re, json, random
from datasets import load_dataset
from . import config as C

# ---------------------------------------------------------------- YAGO decode
def yago_decode_label(url_or_segment: str) -> str:
    if not url_or_segment:
        return ""
    seg = url_or_segment.split("/")[-1]
    seg = re.sub(r"_u([0-9a-fA-F]{4})_", lambda m: chr(int(m.group(1), 16)), seg)
    seg = re.sub(r"_Q\d+$", "", seg)                     # strip trailing Wikidata QID
    seg = seg.replace("_", " ")
    return re.sub(r"\s+", " ", seg).strip()

SLOT_ORDER = ["nationality", "birthplace", "university", "birthdate", "email", "occupation", "uuid"]
META_KEYS = {"nationality": "nationality", "birthplace": "birthplace", "university": "alumni_of",
             "birthdate": "birthdate", "email": "email", "occupation": "occupation", "uuid": "uuid"}
NEEDS_DECODE = {"nationality", "birthplace", "university", "occupation"}
LEAD_INS = {
    "nationality": lambda n: f"{n} is from ",
    "birthplace":  lambda n: f"{n} was born in ",
    "university":  lambda n: f"{n} is an alumni of ",
    "birthdate":   lambda n: f"{n} was born on ",
    "email":       lambda n: f"{n} receives email at ",
    "occupation":  lambda n: f"{n} is a ",
    "uuid":        lambda n: f"{n} has the unique identifier ",
}

def get_slot_value(meta: dict, slot: str) -> str:
    raw = meta[META_KEYS[slot]]
    return yago_decode_label(raw) if slot in NEEDS_DECODE else raw

def sentence(slot, name, value):
    return LEAD_INS[slot](name) + value + "."

# ---------------------------------------------------------------- leak-free prompts
def build_prefixes(meta, target_slot):
    """3 prefix-only formats; NONE contains target_slot's value."""
    name = meta["full_name"]
    idx = SLOT_ORDER.index(target_slot)
    parts = [sentence(s, name, get_slot_value(meta, s)) for s in SLOT_ORDER[:idx]]
    full_prefix = (" ".join(parts) + " " if parts else "") + LEAD_INS[target_slot](name)
    if target_slot == "nationality":
        intro_prefix = LEAD_INS[target_slot](name)
    else:
        intro_prefix = sentence("nationality", name, get_slot_value(meta, "nationality")) + " " + LEAD_INS[target_slot](name)
    return {"full_prefix": full_prefix, "intro_prefix": intro_prefix, "name_only": LEAD_INS[target_slot](name)}

def build_full_suffix(meta, target_slot):
    name = meta["full_name"]
    idx = SLOT_ORDER.index(target_slot)
    return " ".join(sentence(s, name, get_slot_value(meta, s)) for s in SLOT_ORDER[idx + 1:])

TEMPLATE_FORMATS = ["name_only", "intro_prefix", "full_prefix", "full_prefix_full_suffix"]

# ---------------------------------------------------------------- splits
def load_yago_splits(seed=C.SEED):
    """Unseen from TEST (dup=0); Unlearn/Keep = seeded-shuffle halves of TRAIN dup=256."""
    train = load_dataset(C.YAGO_DATASET, split="train")
    test  = load_dataset(C.YAGO_DATASET, split="test")
    dup256 = train.filter(lambda x: x["duplicates"] == 256)
    idx = list(range(len(dup256)))
    random.Random(seed).shuffle(idx)
    half = len(idx) // 2
    forget_set = dup256.select(idx[:half])
    keep_set   = dup256.select(idx[half:])
    dup0 = test.filter(lambda x: x["duplicates"] == 0)
    u_idx = list(range(len(dup0)))
    random.Random(seed).shuffle(u_idx)
    unseen_set = dup0.select(u_idx[:C.UNSEEN_SUBSAMPLE])
    print(f"[data] Unlearn={len(forget_set)} Keep={len(keep_set)} Unseen={len(unseen_set)}")
    return forget_set, keep_set, unseen_set

def load_retain_texts():
    rt = load_dataset(C.RETAIN_DATASET, C.RETAIN_CONFIG, split="train")
    rt = rt.filter(lambda x: len(x["text"].strip()) > C.RETAIN_MIN_LEN)
    print(f"[data] retain (WikiText): {len(rt)} rows")
    return rt

def build_distractor_pool(forget_set, keep_set, unseen_set, target_slot):
    """sorted(set) over the combined corpus (deterministic across runs)."""
    vals = set()
    for ds in (forget_set, keep_set, unseen_set):
        for r in ds:
            vals.add(get_slot_value(json.loads(r["meta"]), target_slot))
    return sorted(vals)

def build_infill_examples(dataset, target_slot, template_format, distractor_pool, seed=C.SEED):
    rng = random.Random(seed)
    n_cand = C.N_CANDIDATES_EMAIL if target_slot == "email" else C.N_CANDIDATES
    out = []
    for row in dataset:
        meta = json.loads(row["meta"])
        true_value = get_slot_value(meta, target_slot)
        prefixes = build_prefixes(meta, target_slot)
        pool = [v for v in distractor_pool if v != true_value]
        distractors = rng.sample(pool, min(n_cand - 1, len(pool)))
        candidates = distractors + [true_value]
        rng.shuffle(candidates)
        ex = {"prefix": prefixes["full_prefix"] if template_format == "full_prefix_full_suffix" else prefixes[template_format],
              "candidates": candidates, "true_index": candidates.index(true_value)}
        if template_format == "full_prefix_full_suffix":
            ex["suffix"] = build_full_suffix(meta, target_slot)
        out.append(ex)
    return out
