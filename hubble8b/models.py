"""Model loading / movement + Hugging Face I/O (models and artifacts).

Loading rule (§7 of MASTER_PROMPT_8B): never device_map="auto"; pin one device;
assert single-device after load. HF token comes from the environment only.
"""
import os, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import config as C

_tokenizer = None

def hf_login():
    if not C.HF_TOKEN:
        print("[hf] no HF_TOKEN in env — public models still load; push/upload will fail.")
        return False
    from huggingface_hub import login
    login(token=C.HF_TOKEN)
    print("[hf] logged in.")
    return True

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(C.MODEL_PERTURBED, revision=C.REVISION)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "right"
    return _tokenizer

def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(name_or_path, device="cpu", revision=None):
    """Single source of truth for every model load. No device_map (silently breaks .to())."""
    m = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=C.DTYPE, revision=revision)
    m.to(device)
    m.eval()
    assert not hasattr(m, "hf_device_map"), f"{name_or_path} unexpectedly has hf_device_map"
    devs = {str(p.device) for p in m.parameters()}
    assert len(devs) == 1, f"{name_or_path} split across devices: {devs}"
    return m

def load_named(name, device=C.DEVICE):
    """Resolve a logical model name to a checkpoint. Baselines from base repos with
    REVISION; unlearned methods from their HF model repo (or a local dir if present)."""
    if name == "Standard":  return load_model(C.MODEL_STANDARD,  device, revision=C.REVISION)
    if name == "Perturbed": return load_model(C.MODEL_PERTURBED, device, revision=C.REVISION)
    method = name.lower()
    local = os.path.join(C.workdir(), f"{method}_unlearned_checkpoint")
    if os.path.isdir(local):
        return load_model(local, device)
    return load_model(C.MODEL_REPO_TMPL.format(method=method), device)

def to_gpu(model, device=C.DEVICE):
    model.to(device); return model

def to_cpu(model):
    model.to("cpu"); free_cuda(); return model

def make_optimizer(params, lr):
    """8-bit AdamW when available (memory); same optimizer across all methods for parity."""
    if C.USE_8BIT_ADAM:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr)
        except Exception as e:
            print(f"[opt] bitsandbytes unavailable ({e}); using torch AdamW.")
    return torch.optim.AdamW(params, lr=lr)

# ---------------------------------------------------------------- HF push / upload
def _ensure_public(repo_id, repo_type=None):
    """Best-effort: make the repo public. create_repo(exist_ok=True) will NOT flip an
    already-existing private repo, so set visibility explicitly (API name varies by
    huggingface_hub version)."""
    from huggingface_hub import HfApi
    api = HfApi()
    for fn in ("update_repo_settings", "update_repo_visibility"):
        try:
            getattr(api, fn)(repo_id=repo_id, repo_type=repo_type, private=False)
            return True
        except Exception:
            continue
    print(f"[hf] could not confirm public visibility for {repo_id} (check manually).")
    return False

def push_model(model, tokenizer, method):
    repo = C.MODEL_REPO_TMPL.format(method=method)
    from huggingface_hub import create_repo
    create_repo(repo, exist_ok=True, private=False)
    _ensure_public(repo)
    model.push_to_hub(repo); tokenizer.push_to_hub(repo)
    print(f"[hf] pushed model (public) -> https://huggingface.co/{repo}")
    return repo

def upload_artifact(local_path, path_in_repo=None):
    """Upload a csv/png/json to the shared (public) results dataset repo."""
    from huggingface_hub import HfApi, create_repo
    create_repo(C.RESULTS_REPO, repo_type="dataset", exist_ok=True, private=False)
    _ensure_public(C.RESULTS_REPO, repo_type="dataset")
    HfApi().upload_file(path_or_fileobj=local_path,
                        path_in_repo=path_in_repo or os.path.basename(local_path),
                        repo_id=C.RESULTS_REPO, repo_type="dataset")
    print(f"[hf] uploaded {os.path.basename(local_path)} -> datasets/{C.RESULTS_REPO}")

def download_artifact(filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C.RESULTS_REPO, filename, repo_type="dataset")
