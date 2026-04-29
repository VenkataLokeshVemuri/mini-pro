"""
T2I EVALUATOR DASHBOARD v5.0
Pairs with pipeline_v6.py  (Unified Multimodal T2I Evaluation Framework).

CHANGES:
  - Only Score Image tab retained; all other tabs removed
  - latent_bank key removed from JSON output display

RUN:
  python dashboard_v6.py
  python dashboard_v6.py --model-file ./pipeline_v5_outputs/models/evaluator_v1.pt
  python dashboard_v6.py --model-dir ./pipeline_v5_outputs/models
  python dashboard_v6.py --output-dir ./pipeline_v5_outputs
  python dashboard_v6.py --port 7861 --share
  python dashboard_v6.py --no-gpu
"""

# -- Windows UTF-8 fix --------------------------------------------------------
import sys, os
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
# -----------------------------------------------------------------------------

import gc, time, json, logging, argparse, warnings, tempfile, re
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger("dashboard_v6")

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torchvision import transforms, models
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import open_clip; OPEN_CLIP_OK = True
except ImportError:
    OPEN_CLIP_OK = False

try:
    from transformers import (CLIPProcessor, CLIPModel,
                               BlipProcessor, BlipForImageTextRetrieval)
    HF_TRANS_OK = True
except ImportError:
    HF_TRANS_OK = False

def _blip_itm_forward_dash(model, proc, images, texts, device):
    """Version-safe BLIP-ITM wrapper for dashboard (handles all transformers versions)."""
    import torch, numpy as np
    imgs = [im.convert("RGB") if im.mode != "RGB" else im for im in images]
    try:
        inp = proc(images=imgs, text=list(texts), return_tensors="pt",
                   padding=True, truncation=True, max_length=64)
    except Exception as e:
        log.warning(f"[BLIP] Processor error: {e}"); return None
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        try:
            out = model(**inp, use_itm_head=True)
            if hasattr(out, "itm_score"):
                sc = torch.softmax(out.itm_score.float(), dim=-1)[:, 1]
                return sc.cpu().numpy().astype(np.float32)
            if hasattr(out, "logits"):
                lg = out.logits.float()
                if lg.dim() == 2 and lg.shape[1] == 2:
                    return torch.softmax(lg, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
                return torch.sigmoid(lg.squeeze(-1)).cpu().numpy().astype(np.float32)
        except TypeError:
            pass
        except Exception as e:
            log.warning(f"[BLIP] strategy1 failed: {e}")
        try:
            out = model(**inp)
            for attr in ("itm_score", "logits", "image_text_matching_score"):
                if not hasattr(out, attr): continue
                lg = getattr(out, attr).float()
                if lg.dim() == 2 and lg.shape[1] == 2:
                    return torch.softmax(lg, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
                return torch.sigmoid(lg.squeeze(-1)).cpu().numpy().astype(np.float32)
        except Exception as e:
            log.warning(f"[BLIP] strategy2 failed: {e}")
        try:
            out = model(**inp, use_itm_head=False)
            for attr in ("itm_score", "logits"):
                if hasattr(out, attr):
                    lg = getattr(out, attr).float()
                    return torch.sigmoid(lg.squeeze(-1)).cpu().numpy().astype(np.float32)
        except Exception as e:
            log.warning(f"[BLIP] strategy3 failed: {e}")
    log.warning("[BLIP] All forward strategies failed")
    return None

try:
    import psutil; PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLT_OK = True
except ImportError:
    PLT_OK = False

try:
    from scipy import linalg as _scipy_linalg
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import spacy as _spacy
    _SPACY_NLP = None
    SPACY_OK = True
except ImportError:
    SPACY_OK = False

try:
    import inspect, gradio as gr
    GRADIO_OK = True
    _ip = set(inspect.signature(gr.Image.__init__).parameters)
    _bp = set(inspect.signature(gr.Button.__init__).parameters)
    _lp = set(inspect.signature(gr.Blocks.launch).parameters)
    GR_IMG_DL   = "show_download_button" in _ip
    GR_BTN_SIZE = "size"     in _bp
    GR_SHOW_API = "show_api" in _lp
    log.info(f"Gradio {gr.__version__}")
except ImportError:
    GRADIO_OK = GR_IMG_DL = GR_BTN_SIZE = GR_SHOW_API = False

# -- constants -----------------------------------------------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# -- spaCy decomposition helper -----------------------------------------------
_REL_PREPS_DASH = {
    "on","in","at","beside","next","near","behind","front","above","below",
    "under","over","between","among","inside","outside","around","against",
}
_ACT_VERBS_DASH = {
    "stand","sit","walk","run","eat","hold","carry","wear","look","ride",
    "fly","jump","sleep","play","drive","park","lean","hang","climb",
}

_DET_WORDS_DASH = {"a", "an", "the", "this", "that", "these", "those"}
_GENERIC_OBJECTS_DASH = {
    "photo", "picture", "image", "scene", "shot", "view",
    "portrait", "frame", "video", "drawing", "painting", "illustration",
    "thing", "object", "item", "stuff", "something",
}

_COLOUR_SYNONYMS_DASH: Dict[str, List[str]] = {
    "red":    ["crimson", "scarlet"],
    "blue":   ["azure", "navy"],
    "green":  ["emerald", "olive"],
    "white":  ["pale", "light"],
    "black":  ["dark", "ebony"],
    "yellow": ["golden", "amber"],
    "brown":  ["tan", "beige"],
    "large":  ["big", "huge"],
    "small":  ["tiny", "little"],
    "old":    ["ancient", "aged"],
    "new":    ["modern", "fresh"],
}

_REL_SWAPS_DASH: Dict[str, str] = {
    "beside": "near", "next":  "close to", "behind": "in front of",
    "above":  "below", "under": "over",    "inside": "outside",
    "on":     "beside", "in":   "near",
}

def _get_nlp_dash():
    global _SPACY_NLP
    if not SPACY_OK: return None
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = _spacy.load("en_core_web_sm")
        except OSError:
            return None
    return _SPACY_NLP


def _normalize_concept_text_dash(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [tok for tok in text.split() if tok not in _DET_WORDS_DASH]
    return " ".join(tokens)


def _concept_query_variants_dash(concept: str, prefix: str) -> List[str]:
    concept = _normalize_concept_text_dash(concept)
    if not concept or concept == "(none)":
        return []

    variants = [concept]
    head = concept.split()[-1]
    if head and head not in variants:
        variants.append(head)

    templates = {
        "a photo of": ["a photo of {}", "a realistic photo of {}",
                        "a clear photo of {}"],
        "a photo with": ["a photo with {}", "a photo containing {}",
                          "an image with {}"],
        "a photo showing": ["a photo showing {}", "a scene showing {}",
                             "an image showing {}"],
    }.get(prefix, [f"{prefix} {{}}", "a photo of {}", "a realistic photo of {}"])

    queries: List[str] = []
    for v in variants[:2]:
        for tmpl in templates:
            q = tmpl.format(v)
            if q not in queries:
                queries.append(q)
            if len(queries) >= 6:
                return queries
    return queries

def decompose_dash(caption: str) -> Dict:
    """Return {objects, attributes, relations} lists for a caption."""
    nlp = _get_nlp_dash()
    if nlp is None:
        words = [_normalize_concept_text_dash(w) for w in caption.split()]
        words = [w for w in words if w and len(w) > 2 and w not in _GENERIC_OBJECTS_DASH]
        return {"objects": words, "attributes": [], "relations": []}
    doc = nlp(caption)
    objects, attributes, relations = [], [], []
    for chunk in doc.noun_chunks:
        phrase = _normalize_concept_text_dash(chunk.text)
        root = chunk.root.lemma_.lower()
        if phrase and phrase not in _GENERIC_OBJECTS_DASH:
            objects.append(phrase)
        if root not in _DET_WORDS_DASH and root not in _GENERIC_OBJECTS_DASH:
            objects.append(root)
        for tok in chunk:
            if tok.pos_ == "ADJ":
                attributes.append(tok.lemma_.lower())
    for tok in doc:
        lem = tok.lemma_.lower()
        if tok.pos_ == "ADP" and lem in _REL_PREPS_DASH:
            relations.append(lem)
        if tok.pos_ == "VERB" and lem in _ACT_VERBS_DASH:
            relations.append(lem)
    def _dedup(lst): return list(dict.fromkeys(lst))
    return {
        "objects":    _dedup(objects)    or ["(none)"],
        "attributes": _dedup(attributes) or [],
        "relations":  _dedup(relations)  or [],
    }


def perturb_caption_dash(caption: str, comp: Dict[str, List[str]],
                         n: int = 3, seed: int = 42) -> List[str]:
    import random
    rng = random.Random(seed)
    perturbed: List[str] = []
    text = caption

    for attr in rng.sample(comp.get("attributes", []),
                           min(len(comp.get("attributes", [])), 2)):
        syns = _COLOUR_SYNONYMS_DASH.get(attr.lower(), [])
        if syns:
            syn = rng.choice(syns)
            p = text.replace(attr, syn, 1)
            if p != text and p not in perturbed:
                perturbed.append(p)
            if len(perturbed) >= n:
                return perturbed[:n]

    for rel in rng.sample(comp.get("relations", []),
                          min(len(comp.get("relations", [])), 2)):
        swap = _REL_SWAPS_DASH.get(rel.lower())
        if swap:
            p = text.replace(rel, swap, 1)
            if p != text and p not in perturbed:
                perturbed.append(p)
            if len(perturbed) >= n:
                return perturbed[:n]

    if len(perturbed) < n:
        prefixes = ["An image showing", "A picture of", "A photo depicting"]
        for pf in prefixes:
            p = f"{pf} {text[0].lower()}{text[1:]}" if len(text) > 1 else f"{pf} {text}"
            if p not in perturbed:
                perturbed.append(p)
            if len(perturbed) >= n:
                break

    return perturbed[:n]

# -- CLIP zero-shot presence scorer -------------------------------------------
def clip_presence_score(engine_ref, img: "Image.Image",
                        concepts: List[str], prefix: str = "a photo of") -> float:
    if not concepts or concepts == ["(none)"]:
        return 0.5
    if not OPEN_CLIP_OK or engine_ref._clip is None:
        return float("nan")
    import torch, torch.nn.functional as _F
    groups = []
    for c in concepts[:8]:
        q = _concept_query_variants_dash(c, prefix)
        if q:
            groups.append(q)
    queries = [q for group in groups for q in group]
    if not queries:
        return 0.5
    try:
        tok  = engine_ref._oc_tok(queries).to(engine_ref.device)
        with torch.no_grad():
            tf   = engine_ref._clip.encode_text(tok)
            it   = engine_ref._clip_prep(img.convert("RGB")).unsqueeze(0).to(engine_ref.device)
            imf  = engine_ref._clip.encode_image(it)
            tf   = _F.normalize(tf.float(), dim=-1)
            imf  = _F.normalize(imf.float(), dim=-1)
            sims = (tf @ imf.T).squeeze(-1).cpu().numpy()
        idx = 0
        concept_scores = []
        for group in groups:
            g = sims[idx:idx + len(group)]
            if len(g):
                concept_scores.append(float(np.max(g)))
            idx += len(group)
        return float(np.mean(concept_scores)) if concept_scores else 0.5
    except Exception as e:
        log.warning(f"[presence] {e}")
        return float("nan")


_REF_FID_CACHE: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}


def _fid(real: np.ndarray, gen: np.ndarray, eps: float = 1e-6) -> Tuple[float, str]:
    if not SCIPY_OK:
        raise RuntimeError("scipy is required for FID")
    warn = ""
    n_real, n_gen = len(real), len(gen)
    FID_MIN_SAMPLES = 50
    if n_real < FID_MIN_SAMPLES or n_gen < FID_MIN_SAMPLES:
        return -1.0, f"n_real={n_real}, n_gen={n_gen} (min {FID_MIN_SAMPLES})"
    if n_real < 100 or n_gen < 100:
        warn = f"low_n(r={n_real},g={n_gen})"
    if n_real < 512 or n_gen < 512:
        if warn:
            warn += " "
        warn += "recommend_n>=512"
    try:
        mu_r, mu_g = real.mean(0), gen.mean(0)
        sr = np.cov(real, rowvar=False) + np.eye(real.shape[1]) * eps
        sg = np.cov(gen,  rowvar=False) + np.eye(gen.shape[1])  * eps
        cond_r = np.linalg.cond(sr)
        cond_g = np.linalg.cond(sg)
        if cond_r > 1e12 or cond_g > 1e12:
            return -1.0, f"singular_cov(cond_r={cond_r:.2e})"
        diff = mu_r - mu_g
        cm, _ = _scipy_linalg.sqrtm(sr @ sg, disp=False)
        if np.iscomplexobj(cm):
            mx = np.abs(np.diag(cm).imag).max()
            if mx > 1e-3:
                warn += f" imag={mx:.4e}" if warn else f"imag={mx:.4e}"
                log.warning(f"[FID] Matrix sqrt returned complex values (imag_max={mx:.4e})")
            cm = cm.real
        raw = float(diff @ diff + np.trace(sr + sg - 2 * cm))
        if np.isnan(raw) or np.isinf(raw):
            return -1.0, f"nan_or_inf_result({raw:.4e})"
        if raw < 0:
            warn += " clamped" if warn else "clamped"
        return max(raw, 0.0), warn.strip()
    except Exception as e:
        return -1.0, f"error({str(e)[:30]})"


def _kid(real: np.ndarray, gen: np.ndarray,
         subsets: int = 100, ss: int = 1000, deg: int = 3) -> Tuple[float, float]:
    n_r, n_g = real.shape[0], gen.shape[0]
    KID_MIN_SAMPLES = 20
    if n_r < KID_MIN_SAMPLES or n_g < KID_MIN_SAMPLES:
        return -1.0, -1.0
    ss_adj = min(ss, n_r, n_g)
    if ss_adj < 10:
        return -1.0, -1.0
    g = 1.0 / real.shape[1]

    def pk(x, y):
        return (g * (x @ y.T) + 1.0) ** deg

    rng = np.random.default_rng(42)
    vals = []
    try:
        for _ in range(subsets):
            ri = rng.choice(n_r, ss_adj, replace=False)
            gi = rng.choice(n_g, ss_adj, replace=False)
            r = real[ri].astype(np.float64)
            ge = gen[gi].astype(np.float64)
            krr = pk(r, r)
            kgg = pk(ge, ge)
            krg = pk(r, ge)
            kid_val = float(
                (krr.sum() - np.trace(krr)) / (ss_adj * (ss_adj - 1)) +
                (kgg.sum() - np.trace(kgg)) / (ss_adj * (ss_adj - 1)) -
                2 * krg.mean())
            if np.isnan(kid_val) or np.isinf(kid_val):
                continue
            vals.append(kid_val)
    except Exception as e:
        log.warning(f"[KID] Calculation failed: {e}")
        return -1.0, -1.0
    if not vals:
        return -1.0, -1.0
    a = np.array(vals)
    return float(max(a.mean(), 0.0)), float(a.std())


def _is(logits: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    N = logits.shape[0]
    IS_MIN_SAMPLES = max(20, splits * 2)
    if N < IS_MIN_SAMPLES:
        return -1.0, -1.0
    try:
        def sm(x):
            e = np.exp(x - x.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)
        pyx = sm(logits)
        ss = N // splits
        sc = []
        for k in range(splits):
            end = N if k == splits - 1 else (k + 1) * ss
            p = pyx[k * ss:end]
            if len(p) == 0:
                continue
            py = p.mean(0, keepdims=True)
            kl_div = (p * (np.log(p + 1e-10) - np.log(py + 1e-10))).sum(1).mean()
            is_score = float(np.exp(kl_div))
            if np.isnan(is_score) or np.isinf(is_score):
                continue
            sc.append(is_score)
        if not sc:
            return -1.0, -1.0
        a = np.array(sc)
        return float(a.mean()), float(a.std())
    except Exception as e:
        log.warning(f"[IS] Calculation failed: {e}")
        return -1.0, -1.0


def _load_reference_fidelity(output_dir: str, split: str) -> Dict[str, np.ndarray]:
    key = (str(Path(output_dir).resolve()), split)
    if key in _REF_FID_CACHE:
        return _REF_FID_CACHE[key]
    output_path = Path(output_dir)
    new_ref_path = output_path / "reference_stats.npz"
    if new_ref_path.exists():
        log.info(f"[Reference] Loading new reference stats from {new_ref_path}")
        try:
            raw = np.load(new_ref_path, allow_pickle=False)
            mu = raw["mu"].astype(np.float32)
            sigma = raw["sigma"].astype(np.float32)
            n_images = int(raw["n_images"])
            feature_dim = int(raw["feature_dim"])
            log.info(f"[Reference] Using {n_images} real images, {feature_dim}D features")
            ref = {
                "pool": mu.reshape(1, -1),
                "logits": np.zeros((1, 1000), dtype=np.float32),
                "n_real": np.array([n_images], dtype=np.int32),
                "path": np.array([str(new_ref_path)], dtype=object),
                "mu": mu,
                "sigma": sigma,
            }
            _REF_FID_CACHE[key] = ref
            log.info(f"[Reference] ✓ Loaded reference stats (approach: reference_stats.npz)")
            return ref
        except Exception as e:
            log.warning(f"[Reference] Failed to load from {new_ref_path}: {e}")
    p = output_path / f"p2_{split}.npz"
    if p.exists():
        log.info(f"[Reference] Loading from pipeline outputs: {p}")
        try:
            raw = np.load(p, allow_pickle=False)
            label_a = raw["label_a"]
            real_idx = np.where(label_a == 0)[0]
            ref = {
                "pool": raw["inc_pool"][real_idx].astype(np.float32),
                "logits": raw["inc_log"][real_idx].astype(np.float32),
                "n_real": np.array([len(real_idx)], dtype=np.int32),
                "path": np.array([str(p)], dtype=object),
            }
            _REF_FID_CACHE[key] = ref
            log.info(f"[Reference] ✓ Loaded reference from pipeline (approach: p2_{split}.npz)")
            return ref
        except Exception as e:
            log.warning(f"[Reference] Failed to load from {p}: {e}")
    raise FileNotFoundError(
        f"Missing reference embeddings for FID/KID/IS evaluation.\n"
        f"Expected one of:\n"
        f"  1. {new_ref_path} (NEW: run extract_reference_stats.py)\n"
        f"  2. {p} (OLD: from pipeline_v6.py)\n"
    )


C_BG      = "#070b14"
C_SURFACE = "#0e1425"
C_CARD    = "#111827"
C_BORDER  = "#1e2d45"
C_BORDER2 = "#2a3f5e"
C_TEXT    = "#e2eaf4"
C_DIM     = "#7a9bbf"
C_ACCENT  = "#38bdf8"
C_GREEN   = "#34d399"
C_AMBER   = "#fbbf24"
C_RED     = "#f87171"
C_PURPLE  = "#a78bfa"
VER_COL   = {
    "High Alignment":    C_GREEN,
    "Partial Alignment": C_AMBER,
    "Low Alignment":     C_RED,
    "Photorealistic": C_GREEN, "Semi-Realistic": C_AMBER, "AI-Generated": C_RED,
    "High Quality":   C_GREEN, "Medium Quality": C_AMBER, "Low Quality":  C_RED,
}


# ==============================================================================
#  MODEL
# ==============================================================================
class EvaluatorHead(nn.Module):
    def __init__(self, in_dim=10, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(0.25),
            nn.Linear(hidden, hidden//2), nn.LayerNorm(hidden//2),
            nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden//2, hidden//4), nn.GELU(),
            nn.Linear(hidden//4, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)


class EvaluatorHeadV3(nn.Module):
    def __init__(self, in_dim=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden//2, hidden//4), nn.GELU(),
            nn.Linear(hidden//4, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)


class Normalizer:
    ALL_KEYS = ["clip_sim", "blip_score", "neg_fid", "neg_kid", "is_score",
                 "robustness_score"]
    def __init__(self):
        self.active_keys = self.ALL_KEYS
        self.stats: Dict[str, Tuple] = {}
    def transform(self, d: Dict) -> np.ndarray:
        cols = []
        for k in self.active_keys:
            mu, std = self.stats.get(k, (0.0, 1.0))
            raw = d.get(k, None)
            if raw is None:
                cols.append(np.zeros(1, dtype=np.float32))
                continue
            arr = np.asarray(raw, dtype=np.float32)
            z = (arr - mu) / std
            z = np.where(np.isnan(arr) | (arr < -990), 0.0, z)
            cols.append(z)
        return np.stack(cols, axis=1)
    def from_dict(self, d: Dict):
        self.active_keys = d.get("active_keys", self.ALL_KEYS)
        self.stats = {k: tuple(v) for k, v in d.get("stats", {}).items()}


class LoadedModel:
    def __init__(self, device):
        self.device = device; self.head = None
        self.norm = Normalizer(); self.meta: Dict = {}
        self.threshold = 0.5; self.active_keys = Normalizer.ALL_KEYS
        self.loaded = False; self.version = "?"; self.timestamp = "?"
        self.train_history: List[Dict] = []; self.path = ""

    def _heuristic_composite(self, clip_sim: float, blip_score: Optional[float],
                             fid: float, kid: float, is_score: float,
                             robustness_score: Optional[float]) -> float:
        clip_n = max(0.0, min(1.0, (float(clip_sim) + 1.0) / 2.0))
        blip_n = max(0.0, min(1.0, float(blip_score) if blip_score is not None else 0.5))
        fid_n  = 1.0 - max(0.0, min(1.0, float(fid) / 100.0)) if fid is not None and fid >= 0 else 0.5
        kid_n  = 1.0 - max(0.0, min(1.0, float(kid) / 0.10)) if kid is not None and kid >= 0 else 0.5
        is_n   = max(0.0, min(1.0, float(is_score) / 30.0)) if is_score is not None and is_score >= 0 else 0.5
        rob_n  = max(0.0, min(1.0, float(robustness_score))) if robustness_score is not None else 0.5
        score = (
            0.35 * clip_n +
            0.25 * blip_n +
            0.15 * fid_n +
            0.10 * kid_n +
            0.10 * is_n +
            0.05 * rob_n
        )
        return float(max(0.0, min(1.0, score)))

    def load(self, path: str):
        ck  = torch.load(path, map_location=self.device, weights_only=False)
        in_dim = ck.get("in_dim", 5)
        ver    = str(ck.get("version", "3.0"))
        Head   = EvaluatorHead if ver.startswith("4") else EvaluatorHeadV3
        self.head = Head(in_dim=in_dim).to(self.device)
        try:
            self.head.load_state_dict(ck["state_dict"])
        except RuntimeError:
            Head2 = EvaluatorHeadV3 if ver.startswith("4") else EvaluatorHead
            self.head = Head2(in_dim=in_dim).to(self.device)
            self.head.load_state_dict(ck["state_dict"])
        self.head.eval()
        self.norm.from_dict(ck["normalizer"])
        self.meta          = ck.get("meta", {})
        self.threshold     = ck.get("threshold", 0.5)
        self.active_keys   = ck.get("active_keys", Normalizer.ALL_KEYS)
        self.version       = ver
        self.timestamp     = ck.get("timestamp", "?")
        self.train_history = ck.get("train_history", [])
        self.loaded        = True
        self.path          = path
        log.info(f"Model v{self.version} loaded: {Path(path).name}")

    @torch.no_grad()
    def predict(self, clip_sim: float, blip_score: Optional[float],
                fid=-1.0, kid=-1.0, is_score=-1.0,
                robustness_score=None) -> Dict:
        if not self.loaded: raise RuntimeError("No model loaded")
        bs   = float(blip_score)       if blip_score is not None else np.nan
        ks   = float(kid)              if kid >= 0               else np.nan
        is_  = float(is_score)         if is_score > 0           else np.nan
        ro   = float(robustness_score) if robustness_score is not None else np.nan
        d   = {k: v for k, v in {
            "clip_sim":         np.array([clip_sim]),
            "blip_score":       np.array([bs]),
            "neg_fid":          np.array([-fid]),
            "neg_kid":          np.array([-ks] if not np.isnan(ks) else [np.nan]),
            "is_score":         np.array([is_]),
            "robustness_score": np.array([ro]),
        }.items() if k in self.active_keys}
        x = self.norm.transform(d)
        learned_p = self.head(torch.tensor(x, dtype=torch.float32)
                              .to(self.device)).item()
        p = self._heuristic_composite(
            clip_sim=clip_sim,
            blip_score=bs if not np.isnan(bs) else None,
            fid=fid,
            kid=ks if not np.isnan(ks) else -1.0,
            is_score=is_ if not np.isnan(is_) else -1.0,
            robustness_score=ro if not np.isnan(ro) else None,
        )
        if   p > 0.70: verdict = "High Alignment"
        elif p > 0.45: verdict = "Partial Alignment"
        else:          verdict = "Low Alignment"
        cs_safe = float(clip_sim) if (isinstance(clip_sim, (int, float))
                        and -2.0 <= float(clip_sim) <= 2.0) else 0.0
        cs_safe = round(max(-1.0, min(1.0, cs_safe)), 4)
        return {"composite_score":  round(p, 4),
                "learned_score":    round(float(learned_p), 4),
                "confidence":       round(abs(p - 0.5)*2, 4),
                "verdict":          verdict,
                "clip_score":       cs_safe,
                "blip_score":       round(bs, 4) if not np.isnan(bs) else None,
                "fid":              round(fid, 4) if fid >= 0         else None,
                "kid":              round(ks, 4)  if not np.isnan(ks) else None,
                "is_score":         round(is_, 4) if not np.isnan(is_) else None,
                "robustness_score": round(ro, 4)  if not np.isnan(ro) else None,
                "threshold":        round(self.threshold, 4)}

    @torch.no_grad()
    def predict_clip_blip_only(self, clip_sim: float,
                               blip_score: Optional[float]) -> Dict:
        cs = float(clip_sim) if isinstance(clip_sim, (int, float)) else 0.0
        cs = max(-1.0, min(1.0, cs))
        bs = float(blip_score) if blip_score is not None else np.nan
        p = self._heuristic_composite(
            clip_sim=cs,
            blip_score=bs if not np.isnan(bs) else None,
            fid=-1.0, kid=-1.0, is_score=-1.0, robustness_score=None,
        )
        if   p > 0.70: verdict = "High Alignment"
        elif p > 0.45: verdict = "Partial Alignment"
        else:          verdict = "Low Alignment"
        return {
            "composite_score":  round(float(p), 4),
            "confidence":       round(abs(float(p) - 0.5) * 2, 4),
            "verdict":          verdict,
            "clip_score":       round(cs, 4),
            "blip_score":       round(bs, 4) if not np.isnan(bs) else None,
            "fid":              None, "kid": None, "is_score": None,
            "robustness_score": None,
            "threshold":        round(self.threshold, 4),
        }


# ==============================================================================
#  INFERENCE ENGINE
# ==============================================================================
class InferenceEngine:
    def __init__(self, device):
        self.device = device; self._clip = None
        self._oc_tok = None; self._hf_proc = None
        self._clip_prep = transforms.Compose([
            transforms.Resize(224,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD)])
        self._inc = None; self._inc_prep = transforms.Compose([
            transforms.Resize((299, 299),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self._ref_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        self._blip = None; self._blip_proc = None
        self._blip_tried = False; self._model = LoadedModel(device)

    def load_model(self, path: str): self._model.load(path)

    def _load_inception(self):
        if self._inc is not None:
            return
        if not TORCH_OK:
            raise RuntimeError("pip install torch torchvision")
        log.info("Loading InceptionV3 ...")
        inc = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        inc.aux_logits = False; inc.AuxLogits = None
        inc.eval().to(self.device)
        self._inc = inc
        log.info("InceptionV3 ready")

    def _load_reference_fid(self, output_dir: str, split: str) -> Dict[str, np.ndarray]:
        key = (str(Path(output_dir).resolve()), split)
        if key in self._ref_cache:
            return self._ref_cache[key]
        ref = _load_reference_fidelity(output_dir, split)
        self._ref_cache[key] = ref
        return ref

    @torch.no_grad()
    def _inception_features(self, images: List[Image.Image], batch_size: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        self._load_inception()
        imgs = [im.convert("RGB") if im.mode != "RGB" else im for im in images]
        if not imgs:
            return np.empty((0, 2048), dtype=np.float32), np.empty((0, 1000), dtype=np.float32)
        bs = batch_size or (16 if self.device.type == "cuda" else 8)
        pool_parts: List[np.ndarray] = []
        logit_parts: List[np.ndarray] = []
        for start in range(0, len(imgs), bs):
            chunk = imgs[start:start + bs]
            pool_buf: List[np.ndarray] = []
            def _hook(_m, _inp, out):
                pool_buf.append(out.squeeze(-1).squeeze(-1).detach().cpu().float().numpy())
            handle = self._inc.avgpool.register_forward_hook(_hook)
            try:
                ts = [self._inc_prep(img) for img in chunk]
                t = torch.stack(ts).to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    logits = self._inc(t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                if not pool_buf:
                    raise RuntimeError("Inception avgpool hook did not capture features")
                pool_parts.append(pool_buf[0])
                logit_parts.append(logits.detach().cpu().float().numpy())
            finally:
                handle.remove()
        return np.vstack(pool_parts), np.vstack(logit_parts)

    def _generated_bank_path(self, output_dir: str, split: str) -> Path:
        return Path(output_dir) / f"generated_latent_bank_{split}.npz"

    def _load_generated_bank(self, output_dir: str, split: str) -> Dict[str, np.ndarray]:
        p = self._generated_bank_path(output_dir, split)
        if not p.exists():
            return {
                "pool": np.empty((0, 2048), dtype=np.float32),
                "logits": np.empty((0, 1000), dtype=np.float32),
                "path": str(p),
            }
        raw = np.load(p, allow_pickle=False)
        pool = raw["inc_pool"].astype(np.float32)
        logits = raw["inc_log"].astype(np.float32)
        if pool.ndim != 2 or pool.shape[1] != 2048:
            raise RuntimeError(f"Invalid bank pool shape: {pool.shape} ({p})")
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError(f"Invalid bank logits shape: {logits.shape} ({p})")
        return {"pool": pool, "logits": logits, "path": str(p)}

    def _save_generated_bank(self, output_dir: str, split: str,
                             pool: np.ndarray, logits: np.ndarray) -> str:
        p = self._generated_bank_path(output_dir, split)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez_compressed(tmp, inc_pool=pool.astype(np.float32),
                            inc_log=logits.astype(np.float32))
        os.replace(str(tmp), str(p))
        return str(p)

    def update_latent_bank_metrics(self, image: Image.Image,
                                   output_dir: str,
                                   split: str = "test") -> Dict[str, Any]:
        ref = None
        ref_error = None
        try:
            ref = self._load_reference_fid(output_dir, split)
        except FileNotFoundError as e:
            ref_error = str(e)
            log.warning(f"[Latent Bank] Reference stats not found: {ref_error}")

        pool_new, log_new = self._inception_features([image])
        bank = self._load_generated_bank(output_dir, split)

        if bank["pool"].size:
            gen_pool = np.vstack([bank["pool"], pool_new])
            gen_log = np.vstack([bank["logits"], log_new])
        else:
            gen_pool = pool_new
            gen_log = log_new

        bank_path = self._save_generated_bank(output_dir, split, gen_pool, gen_log)
        n_gen = gen_pool.shape[0]

        if ref is None:
            fid_v, fid_warn = -1.0, "reference_not_loaded"
            kid_m, kid_s = -1.0, -1.0
            is_m, is_s = -1.0, -1.0
        else:
            fid_v, fid_warn = _fid(ref["pool"], gen_pool)
            kid_m, kid_s = _kid(ref["pool"], gen_pool)
            is_m, is_s = _is(gen_log)

        return {
            "n_generated": int(n_gen),
            "fid": round(fid_v, 4) if fid_v >= 0 else None,
            "kid_mean": round(kid_m, 4) if kid_m >= 0 else None,
            "is_mean": round(is_m, 4) if is_m >= 0 else None,
            "fid_warn": fid_warn,
        }

    def _load_clip(self):
        if self._clip: return
        if OPEN_CLIP_OK:
            m, _, prep = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=self.device)
            m.eval(); self._clip = m
            self._oc_tok  = open_clip.get_tokenizer("ViT-B-32")
            self._clip_prep = prep
            log.info("CLIP loaded via open_clip (ViT-B-32/openai)")
        elif HF_TRANS_OK:
            hf = "openai/clip-vit-base-patch32"
            self._clip    = CLIPModel.from_pretrained(hf).to(self.device).eval()
            self._hf_proc = CLIPProcessor.from_pretrained(hf)
            log.info(f"CLIP loaded via HF transformers ({hf})")
        else:
            raise RuntimeError("pip install open_clip_torch or transformers")

    @torch.no_grad()
    def clip_score(self, prompt: str, image: Image.Image) -> float:
        self._load_clip()
        img = image.convert("RGB") if image.mode != "RGB" else image
        if self._oc_tok:
            it  = self._clip_prep(img).unsqueeze(0).to(self.device)
            tok = self._oc_tok([prompt]).to(self.device)
            tf  = self._clip.encode_text(tok)
            imf = self._clip.encode_image(it)
            tf  = F.normalize(tf.float(), dim=-1)
            imf = F.normalize(imf.float(), dim=-1)
        else:
            enc = self._hf_proc(
                text=[prompt], images=[img],
                return_tensors="pt", padding=True,
                truncation=True, max_length=77)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            tf  = self._clip.get_text_features(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"))
            imf = self._clip.get_image_features(
                pixel_values=enc["pixel_values"])
            tf  = F.normalize(tf.float(), dim=-1)
            imf = F.normalize(imf.float(), dim=-1)
        sim = (tf * imf).sum(dim=-1).squeeze()
        score = float(sim.item())
        return max(-1.0, min(1.0, score))

    def _load_blip(self) -> bool:
        if self._blip: return True
        if self._blip_tried: return False
        self._blip_tried = True
        if not HF_TRANS_OK: return False
        candidates = [
            ("Salesforce/blip-itm-base-coco",  "refs/pr/6"),
            ("Salesforce/blip-itm-base-coco",  "main"),
            ("Salesforce/blip-itm-large-coco", "refs/pr/6"),
            ("Salesforce/blip-itm-large-coco", "main"),
        ]
        for mn, rev in candidates:
            try:
                log.info(f"[BLIP] Trying {mn} @ {rev} ...")
                kw = {"revision": rev} if rev != "main" else {}
                proc = BlipProcessor.from_pretrained(mn, **kw)
                try:
                    model = BlipForImageTextRetrieval.from_pretrained(mn, **kw)
                except Exception as bin_err:
                    err_s = str(bin_err)
                    if "CVE" in err_s or "weights_only" in err_s or "torch.load" in err_s:
                        log.warning(f"[BLIP] torch.load blocked; retrying with weights_only=False ...")
                        import torch as _torch
                        _orig = _torch.load
                        def _patched_load(f, *a, **kw2):
                            kw2["weights_only"] = False
                            return _orig(f, *a, **kw2)
                        _torch.load = _patched_load
                        try:
                            model = BlipForImageTextRetrieval.from_pretrained(mn, **kw)
                        finally:
                            _torch.load = _orig
                    else:
                        raise
                model = model.to(self.device).eval()
                from PIL import Image as _Im
                _res = _blip_itm_forward_dash(model, proc,
                    [_Im.new("RGB", (64, 64))], ["test"], self.device)
                if _res is None:
                    log.warning(f"[BLIP] Smoke test failed for {mn}@{rev}")
                    del model; continue
                log.info(f"[BLIP] Ready: {mn}@{rev}  smoke={_res[0]:.4f}")
                self._blip_proc = proc; self._blip = model
                return True
            except Exception as e:
                log.warning(f"[BLIP] Load failed [{mn}@{rev}]: {e}")
        log.warning("[BLIP] All candidates failed -- BLIP disabled")
        return False

    def blip_score(self, prompt: str, image: Image.Image) -> Optional[float]:
        if not self._load_blip(): return None
        img = image.convert("RGB") if image.mode != "RGB" else image
        res = _blip_itm_forward_dash(self._blip, self._blip_proc,
                                     [img], [prompt], self.device)
        if res is None or len(res) == 0: return None
        return float(res[0])

    def _input_semantic_scores(self, prompt: str, image: Image.Image,
                               base_clip: float) -> Dict[str, Optional[float]]:
        comp = decompose_dash(prompt)
        perturbs = perturb_caption_dash(prompt, comp, n=3, seed=42)
        sims = [float(base_clip)]
        for p in perturbs:
            try:
                sims.append(float(self.clip_score(p, image)))
            except Exception as e:
                log.warning(f"[robustness] perturbation failed: {e}")
        rob_std = float(np.std(sims)) if sims else 0.0
        robust = float(max(0.0, 1.0 - rob_std / 0.3))
        return {"robustness_score": round(robust, 4)}

    def score(self, prompt: str, image: Image.Image,
              ref_fid=-1.0, ref_kid=-1.0, ref_is=-1.0,
              output_dir: Optional[str] = None,
              latent_split: str = "test",
              use_latent_bank: bool = True) -> Dict:
        t0 = time.time()
        cs = self.clip_score(prompt, image)
        if not (-1.0 <= cs <= 1.0):
            log.error(f"[CLIP] Out-of-range score {cs:.4f} -- clamping to [-1,1].")
            cs = max(-1.0, min(1.0, cs))
        log.info(f"[Score] CLIP={cs:.4f}")
        bs = self.blip_score(prompt, image)
        bs_safe = round(float(bs), 4) if bs is not None else None
        log.info(f"[Score] BLIP={bs_safe}")
        sem = self._input_semantic_scores(prompt, image, cs)
        fid_in = float(ref_fid) if isinstance(ref_fid, (int, float)) and ref_fid >= 0 else -1.0
        kid_in = float(ref_kid) if isinstance(ref_kid, (int, float)) and ref_kid >= 0 else -1.0
        is_in  = float(ref_is)  if isinstance(ref_is,  (int, float)) and ref_is  > 0 else -1.0

        if use_latent_bank and output_dir:
            try:
                latent_metrics = self.update_latent_bank_metrics(
                    image=image, output_dir=output_dir, split=latent_split)
                if latent_metrics.get("fid") is not None:
                    fid_in = float(latent_metrics["fid"])
                if latent_metrics.get("kid_mean") is not None:
                    kid_in = float(latent_metrics["kid_mean"])
                if latent_metrics.get("is_mean") is not None:
                    is_in = float(latent_metrics["is_mean"])
                log.info(
                    "[Score] Latent bank updated: split=%s n_gen=%s FID=%s KID=%s IS=%s",
                    latent_split,
                    latent_metrics.get("n_generated"),
                    latent_metrics.get("fid"),
                    latent_metrics.get("kid_mean"),
                    latent_metrics.get("is_mean"),
                )
            except Exception as e:
                log.warning(f"[Score] Latent bank update failed: {e}")

        r = self._model.predict(
            cs, bs,
            fid=fid_in, kid=kid_in, is_score=is_in,
            robustness_score=sem.get("robustness_score"),
        )
        # Also compute the CLIP+BLIP-only composite (user requested)
        try:
            cb = self._model.predict_clip_blip_only(cs, bs)
            r["clip_blip_composite"] = float(cb.get("composite_score", None))
            r["clip_blip_confidence"] = float(cb.get("confidence", 0.0))
        except Exception:
            r["clip_blip_composite"] = None
            r["clip_blip_confidence"] = None
        # Expose explicit metric fields for downstream charts/UI
        r["fid"] = round(fid_in, 4) if fid_in >= 0 else None
        r["kid"] = round(kid_in, 4) if kid_in >= 0 else None
        r["is_score"] = round(is_in, 4) if is_in >= 0 else None
        r["robustness_score"] = sem.get("robustness_score")
        # NOTE: latent_bank is intentionally NOT added to the result dict
        r["elapsed_sec"] = round(time.time() - t0, 2)
        r["prompt"] = prompt[:100]
        return r


# ==============================================================================
#  CHARTS
# ==============================================================================
def _tmp() -> str:
    return tempfile.NamedTemporaryFile(suffix=".png", delete=False).name


def _style(ax):
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_DIM, labelsize=8.5)
    for sp in ax.spines.values(): sp.set_color(C_BORDER2)
    ax.grid(color=C_BORDER, linewidth=0.5, alpha=0.6)


def chart_gauge(score: float, verdict: str) -> Optional[str]:
    if not PLT_OK: return None
    col = VER_COL.get(verdict, C_DIM)
    fig = plt.figure(figsize=(4.6, 2.8), facecolor=C_BG)
    ax  = fig.add_axes([0.04, 0.04, 0.92, 0.92],
                       polar=True, facecolor=C_CARD)
    t   = np.linspace(np.pi, 0, 300)
    ax.plot(t, np.ones(300)*0.88, color=C_BORDER2,
            linewidth=22, solid_capstyle="round", alpha=0.5)
    end = np.pi - score * np.pi
    sc  = np.linspace(np.pi, max(end, np.pi*0.005), 300)
    ax.plot(sc, np.ones_like(sc)*0.88, color=col,
            linewidth=22, solid_capstyle="round", alpha=0.92)
    ax.plot(sc, np.ones_like(sc)*0.88, color=col,
            linewidth=30, solid_capstyle="round", alpha=0.07)
    for tv, lbl in [(0,".0"),(0.25,".25"),(0.5,".5"),(0.75,".75"),(1.0,"1")]:
        t2 = np.pi - tv * np.pi
        ax.plot([t2, t2], [0.76, 0.81], color=C_BORDER2, linewidth=1.2)
        ax.text(t2, 0.65, lbl, ha="center", va="center",
                fontsize=7.5, color=C_DIM, fontfamily="monospace")
    ax.text(np.pi/2, 0.30, f"{score:.3f}",
            ha="center", va="center", fontsize=32,
            color=col, fontweight="bold", fontfamily="monospace")
    ax.text(np.pi/2, 0.07, verdict.upper(),
            ha="center", va="center", fontsize=9,
            color=col, fontweight="bold", fontfamily="monospace")
    ax.set_ylim(0, 1.1); ax.set_xlim(0, np.pi)
    ax.set_theta_zero_location("W"); ax.set_theta_direction(-1)
    ax.set_yticklabels([]); ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False); ax.grid(False)
    path = _tmp(); plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close(); return path


def chart_radar(result: Dict) -> Optional[str]:
    if not PLT_OK: return None
    col = VER_COL.get(result.get("verdict",""), C_DIM)
    try:
        clip_raw = float(result.get("clip_score", 0))
        if not (-2.0 <= clip_raw <= 2.0): clip_raw = 0.0
    except (TypeError, ValueError): clip_raw = 0.0
    clip_norm = max(0.0, min(1.0, (clip_raw + 1.0) / 2.0))
    items = [("CLIP", clip_norm)]
    if result.get("blip_score") is not None:
        items.append(("BLIP",    result["blip_score"]))
    if result.get("robustness_score") is not None:
        items.append(("Robust",  result["robustness_score"]))
    if result.get("fid") is not None and result["fid"] >= 0:
        items.append(("FID",    max(0, 1-result["fid"]/300)))
    if result.get("is_score") is not None and result["is_score"] >= 0:
        items.append(("IS",     min(1.0, result["is_score"]/50)))
    items.append(("Score",   result.get("composite_score", 0)))
    labels = [i[0] for i in items]; vals = [i[1] for i in items]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vp = vals + [vals[0]]; ap = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(4.4, 4.4),
                            subplot_kw=dict(polar=True), facecolor=C_BG)
    ax.set_facecolor(C_CARD)
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(ap, [r]*(N+1), color=C_BORDER, linewidth=0.7, alpha=0.8)
    for a in angles:
        ax.plot([a, a], [0, 1], color=C_BORDER2, linewidth=0.6)
    ax.fill(ap, vp, color=col, alpha=0.13)
    ax.plot(ap, vp, color=col, linewidth=2.2, alpha=0.95)
    for a, v in zip(angles, vals):
        ax.plot(a, v, "o", color=col, markersize=5.5, zorder=5)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color=C_TEXT, fontsize=9.5,
                        fontweight="bold", fontfamily="monospace")
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.spines["polar"].set_color(C_BORDER2); ax.grid(False)
    ax.set_title("Metric Profile", color=C_TEXT, fontsize=10.5,
                 fontweight="bold", pad=16, fontfamily="monospace")
    path = _tmp()
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close(); return path


def chart_bars(result: Dict) -> Optional[str]:
    if not PLT_OK: return None
    try:
        clip_raw = float(result.get("clip_score", 0))
        if not (-2.0 <= clip_raw <= 2.0): clip_raw = 0.0
    except (TypeError, ValueError): clip_raw = 0.0
    clip_bar = max(0.0, min(1.0, (clip_raw + 1.0) / 2.0))
    C_TEAL  = "#2dd4bf"
    C_ROSE  = "#fb7185"
    items = [("CLIP Score", clip_bar, C_ACCENT)]
    if result.get("blip_score") is not None:
        items.append(("BLIP-ITM",  result["blip_score"],    C_PURPLE))
    if result.get("robustness_score") is not None:
        items.append(("Robustness",   result["robustness_score"], C_AMBER))
    if result.get("fid") is not None and result["fid"] >= 0:
        items.append(("FID Score",    max(0, 1-result["fid"]/300), C_ROSE))
    if result.get("is_score") is not None and result["is_score"] >= 0:
        items.append(("IS Score",     min(1.0, result["is_score"]/50), C_ROSE))
    if result.get("kid") is not None and result["kid"] >= 0:
        items.append(("KID Score",    max(0, 1-result["kid"]/0.5), C_ROSE))
    items.append(("Composite", result.get("composite_score",0),
                  VER_COL.get(result.get("verdict",""), C_DIM)))
    labels = [i[0] for i in items]; vals = [i[1] for i in items]
    cols   = [i[2] for i in items]; n = len(labels)
    fig, ax = plt.subplots(figsize=(5.4, max(2.0, n*0.65)), facecolor=C_BG)
    _style(ax)
    y = np.arange(n)
    ax.barh(y, [1]*n, color=C_BORDER, alpha=0.35, height=0.55)
    ax.barh(y, vals,  color=cols, alpha=0.88, height=0.55)
    ax.barh(y, vals,  color=cols, alpha=0.06, height=0.72)
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.text(min(v+0.025, 0.97), i, f"{v:.3f}",
                va="center", color=c, fontsize=10,
                fontweight="bold", fontfamily="monospace")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=C_TEXT, fontsize=9.5,
                        fontfamily="monospace")
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("Score", color=C_DIM, fontsize=8.5)
    ax.tick_params(axis="x", colors=C_DIM)
    ax.set_title("Metric Breakdown", color=C_TEXT, fontsize=10.5,
                 fontweight="bold", fontfamily="monospace")
    plt.tight_layout(pad=1.0)
    path = _tmp()
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close(); return path


def chart_pies(result: Dict) -> Optional[str]:
    """Draw two pies: left = CLIP vs BLIP contributions; right = generalized 6-factor contributions."""
    if not PLT_OK: return None
    try:
        # Helper: normalize components same as _heuristic_composite
        def norm_clip(x):
            try:
                cs = float(x)
            except Exception:
                cs = 0.0
            return max(0.0, min(1.0, (cs + 1.0) / 2.0))
        def norm_blip(x):
            try: return max(0.0, min(1.0, float(x)))
            except Exception: return 0.5
        def norm_fid(x):
            try:
                if x is None or float(x) < 0: return 0.5
                return 1.0 - max(0.0, min(1.0, float(x) / 100.0))
            except Exception:
                return 0.5
        def norm_kid(x):
            try:
                if x is None or float(x) < 0: return 0.5
                return 1.0 - max(0.0, min(1.0, float(x) / 0.10))
            except Exception:
                return 0.5
        def norm_is(x):
            try:
                if x is None or float(x) <= 0: return 0.5
                return max(0.0, min(1.0, float(x) / 30.0))
            except Exception:
                return 0.5
        def norm_rob(x):
            try:
                if x is None: return 0.5
                return max(0.0, min(1.0, float(x)))
            except Exception:
                return 0.5

        clip_n = norm_clip(result.get("clip_score", result.get("clip_sim", 0)))
        blip_n = norm_blip(result.get("blip_score")) if result.get("blip_score") is not None else 0.5

        # CLIP+BLIP pie (weights from heuristic: 0.35, 0.25)
        w_clip, w_blip = 0.35, 0.25
        comp_clip = w_clip * clip_n
        comp_blip = w_blip * blip_n
        sum_cb = max(1e-6, comp_clip + comp_blip)
        labels_cb = ["CLIP", "BLIP"]
        vals_cb = [comp_clip / sum_cb, comp_blip / sum_cb]
        colors_cb = [C_ACCENT, C_PURPLE]

        # Generalized 6-factor contributions
        fid_v = result.get("fid")
        kid_v = result.get("kid")
        is_v = result.get("is_score")
        rob_v = result.get("robustness_score")
        clip_v = result.get("clip_score", result.get("clip_sim", 0))
        blip_v = result.get("blip_score")

        c_clip = norm_clip(clip_v)
        c_blip = norm_blip(blip_v)
        c_fid  = norm_fid(fid_v)
        c_kid  = norm_kid(kid_v)
        c_is   = norm_is(is_v)
        c_rob  = norm_rob(rob_v)

        weights = {
            "CLIP": 0.35, "BLIP": 0.25, "FID": 0.15,
            "KID": 0.10, "IS": 0.10, "Robustness": 0.05,
        }
        comps = {
            "CLIP": weights["CLIP"] * c_clip,
            "BLIP": weights["BLIP"] * c_blip,
            "FID": weights["FID"] * c_fid,
            "KID": weights["KID"] * c_kid,
            "IS": weights["IS"] * c_is,
            "Robustness": weights["Robustness"] * c_rob,
        }
        total = max(1e-6, sum(comps.values()))
        labels_g = list(comps.keys())
        vals_g = [comps[k] / total for k in labels_g]
        cols_g = [C_ACCENT, C_PURPLE, C_ROSE, C_ROSE, C_ROSE, C_AMBER]

        # Draw pies side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.4), facecolor=C_BG)
        # Left pie: CLIP+BLIP
        axs[0].set_facecolor(C_CARD)
        axs[0].pie(vals_cb, labels=[f"{l}: {v*100:.1f}%" for l, v in zip(labels_cb, vals_cb)],
                   colors=colors_cb, startangle=140, wedgeprops={"edgecolor": C_BORDER2})
        axs[0].set_title("CLIP vs BLIP Contribution", color=C_TEXT, fontsize=10.5)

        # Right pie: Generalized contributions
        axs[1].set_facecolor(C_CARD)
        axs[1].pie(vals_g, labels=[f"{l}: {v*100:.1f}%" for l, v in zip(labels_g, vals_g)],
                   colors=cols_g, startangle=140, wedgeprops={"edgecolor": C_BORDER2})
        axs[1].set_title("Generalized 6-Factor Contribution", color=C_TEXT, fontsize=10.5)

        plt.tight_layout()
        path = _tmp()
        plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=C_BG)
        plt.close()
        return path
    except Exception as e:
        log.warning(f"[chart_pies] failed: {e}")
        return None


# ==============================================================================
#  CSS
# ==============================================================================
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
body, .gradio-container {{
    background: {C_BG} !important;
    color: {C_TEXT} !important;
    font-family: 'Syne', sans-serif !important;
}}
.block, .gr-box, .gr-form, .gr-panel {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER2} !important;
    border-radius: 10px !important;
}}
label span, .label-wrap span {{
    color: {C_DIM} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important; font-weight: 500 !important;
    text-transform: uppercase !important; letter-spacing: .08em !important;
}}
input[type="text"], input[type="number"], textarea {{
    background: {C_SURFACE} !important; color: {C_TEXT} !important;
    border: 1px solid {C_BORDER2} !important; border-radius: 7px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}}
input:focus, textarea:focus {{
    border-color: {C_ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,.15) !important; outline: none !important;
}}
button.primary, .primary-btn {{
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important; border-radius: 8px !important; color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: .1em !important; text-transform: uppercase !important;
    transition: opacity .2s, box-shadow .2s !important;
    box-shadow: 0 0 16px rgba(14,165,233,.25) !important;
}}
button.primary:hover {{ opacity: .85 !important; box-shadow: 0 0 22px rgba(14,165,233,.4) !important; }}
button.secondary {{
    background: {C_SURFACE} !important; border: 1px solid {C_BORDER2} !important;
    color: {C_DIM} !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: .08em !important; text-transform: uppercase !important;
    border-radius: 7px !important; transition: border-color .2s, color .2s !important;
}}
button.secondary:hover {{ border-color: {C_ACCENT} !important; color: {C_TEXT} !important; }}
.accordion {{
    background: {C_SURFACE} !important;
    border: 1px solid {C_BORDER} !important; border-radius: 8px !important;
}}
h1, h2, h3 {{ color: {C_TEXT} !important; font-family: 'Syne', sans-serif !important; }}
p, li {{ color: {C_DIM} !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }}
footer {{ display: none !important; }}
"""


# ==============================================================================
#  GRADIO APP  (Score Image only)
# ==============================================================================
def ikw(): return {"show_download_button": False} if GR_IMG_DL else {}
def bkw(): return {"size": "lg"} if GR_BTN_SIZE else {}


def load_metrics_from_csv(output_dir: str, split: str = "test") -> Tuple[float, float, float]:
    try:
        fid_path = Path(output_dir) / f"p4_{split}_fidelity.csv"
        if fid_path.exists():
            df = pd.read_csv(fid_path)
            fid_val = float(df["fid"].mean()) if "fid" in df.columns else -1.0
            if "kid_mean" in df.columns:
                kid_val = float(df["kid_mean"].mean())
            elif "kid" in df.columns:
                kid_val = float(df["kid"].mean())
            else:
                kid_val = -1.0
            if "is_mean" in df.columns:
                is_val = float(df["is_mean"].mean())
            elif "is" in df.columns:
                is_val = float(df["is"].mean())
            else:
                is_val = -1.0
            log.info(f"Loaded metrics from {fid_path.name}: FID={fid_val:.2f}, KID={kid_val:.4f}, IS={is_val:.2f}")
            return fid_val, kid_val, is_val
    except Exception as e:
        log.warning(f"Failed to load metrics from CSV: {e}")
    return -1.0, -1.0, -1.0


def find_free_port(start_port: int, host: str = "127.0.0.1", max_tries: int = 50) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"Cannot find a free port in range {start_port}-{start_port + max_tries - 1}")


def build_dashboard(engine: InferenceEngine, model_dir: str,
                    output_dir: str = "./pipeline_v5_outputs") -> "gr.Blocks":

    # ---- helpers --------------------
    def verdict_html(verdict: str, score: float, conf: float) -> str:
        col  = VER_COL.get(verdict, C_DIM)
        if verdict in ("Photorealistic","High Quality","High Alignment"):        icon = "+"
        elif verdict in ("Semi-Realistic","Medium Quality","Partial Alignment"): icon = "~"
        else:                                                                    icon = "-"
        sub  = {
            "High Alignment":    "Image closely matches prompt content & attributes",
            "Partial Alignment": "Image partially matches — some objects or attributes missing",
            "Low Alignment":     "Image poorly represents the prompt",
            "Photorealistic": "Indistinguishable from real photo",
            "Semi-Realistic": "Partially realistic appearance",
            "AI-Generated":   "Clearly AI art",
            "High Quality":   "High photorealism",
            "Medium Quality": "Moderate photorealism",
            "Low Quality":    "Low photorealism",
        }.get(verdict, "")
        return (f'<div style="background:{C_SURFACE};border:1px solid {col}44;'
                f'border-radius:12px;padding:20px;text-align:center;'
                f'box-shadow:0 0 24px {col}14">'
                f'<div style="font-size:40px;font-weight:800;color:{col};'
                f'font-family:monospace">{score:.3f}</div>'
                f'<div style="font-size:12px;font-weight:700;color:{col};'
                f'font-family:monospace;margin:5px 0;text-transform:uppercase;'
                f'letter-spacing:.12em">[{icon}] {verdict}</div>'
                f'<div style="font-size:11px;color:{C_DIM};font-family:monospace;'
                f'margin-top:3px">{sub}</div>'
                f'<div style="font-size:11px;color:{C_DIM};font-family:monospace">'
                f'Confidence: {conf:.1%}</div></div>')

    def analysis_html(result: Dict) -> str:
        lines = []
        clip = result.get("clip_score", 0)
        try:
            clip = float(clip)
            if not (-2.0 <= clip <= 2.0):
                clip = float("nan")
        except (TypeError, ValueError):
            clip = float("nan")
        if not (clip != clip):
            clip = max(-1.0, min(1.0, clip))
            if   clip > 0.32: lines.append((C_GREEN, f"CLIP={clip:.3f}  strong alignment"))
            elif clip > 0.22: lines.append((C_AMBER, f"CLIP={clip:.3f}  moderate alignment"))
            else:             lines.append((C_RED,   f"CLIP={clip:.3f}  weak alignment"))
        else:
            lines.append((C_DIM, "CLIP=n/a  (score unavailable)"))
        bs = result.get("blip_score")
        if bs is not None:
            lbl = "high" if bs>0.7 else ("partial" if bs>0.4 else "low")
            c   = C_GREEN if bs>0.7 else (C_AMBER if bs>0.4 else C_RED)
            lines.append((c, f"BLIP={bs:.3f}  ITM {lbl}"))
        fid = result.get("fid")
        if fid is not None and fid >= 0:
            lbl = "close to real" if fid<30 else ("moderate" if fid<100 else "far")
            c   = C_GREEN if fid<30 else (C_AMBER if fid<100 else C_RED)
            lines.append((c, f"FID={fid:.1f}  {lbl}"))
        is_ = result.get("is_score")
        if is_: lines.append((C_DIM, f"IS={is_:.2f}"))
        ro = result.get("robustness_score")
        if ro is not None:
            lbl = "stable" if ro > 0.80 else ("moderate" if ro > 0.60 else "unstable")
            c   = C_GREEN if ro > 0.80 else (C_AMBER if ro > 0.60 else C_RED)
            lines.append((c, f"Robustness={ro:.3f}  score {lbl}"))
        lines.append((C_DIM, f"Elapsed: {result.get('elapsed_sec','?')}s"))
        inner = "".join(
            f'<div style="padding:5px 0;border-bottom:1px solid {C_BORDER};'
            f'font-family:monospace;font-size:11px;color:{c}">&gt; {l}</div>'
            for c, l in lines)
        return (f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER2};'
                f'border-radius:10px;padding:14px">'
                f'<div style="color:{C_DIM};font-size:10px;font-family:monospace;'
                f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">'
                f'Analysis</div>{inner}</div>')

    def clean_result_for_display(r: Dict) -> Dict:
        """Remove latent_bank from result before displaying as JSON."""
        return {k: v for k, v in r.items() if k != "latent_bank"}

    HEADER = (
        f'<div style="background:linear-gradient(135deg,{C_SURFACE},{C_CARD});'
        f'border:1px solid {C_BORDER2};border-radius:14px;padding:22px 28px;'
        f'margin-bottom:4px;position:relative;overflow:hidden">'
        f'<div style="position:absolute;top:-30px;right:-30px;width:180px;'
        f'height:180px;background:radial-gradient(circle,{C_ACCENT}18,transparent 70%);'
        f'pointer-events:none"></div>'
        f'<div style="font-size:26px;font-weight:800;color:{C_TEXT};'
        f'font-family:Syne,sans-serif;letter-spacing:-.02em">'
        f'T2I Evaluator v5 '
        f'<span style="background:linear-gradient(90deg,{C_ACCENT},{C_PURPLE});'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
        f'Dashboard</span></div>'
        f'<div style="display:flex;gap:8px;margin-top:8px;align-items:center">'
        f'<span style="background:linear-gradient(135deg,#0c2a3e,#1a1040);'
        f'color:{C_ACCENT};padding:3px 10px;border-radius:20px;border:1px solid {C_BORDER2};'
        f'font-family:monospace;font-size:10px;font-weight:700;letter-spacing:.1em">v5.0</span>'
        f'<span style="color:{C_DIM};font-family:monospace;font-size:11px">'
        f'train=42k fit &nbsp;|&nbsp; val=9k earlystop &nbsp;|&nbsp; test=45k eval only</span></div>'
        f'</div>'
    )

    # ==========================================================================
    with gr.Blocks(title="T2I Evaluator v5.0", css=CSS,
                   theme=gr.themes.Base()) as demo:

        gr.HTML(HEADER)

        # -------------------- Score Image (only tab) --------------------
        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=290):
                image_in  = gr.Image(type="pil", label="Generated Image",
                                     height=270, **ikw())
                prompt_in = gr.Textbox(label="Prompt",
                    placeholder="a photorealistic cat on a beach ...",
                    lines=3)
                with gr.Accordion("Reference Metrics (optional)", open=True):
                    gr.HTML(f'<div style="color:{C_DIM};font-size:11px;'
                            f'font-family:monospace;padding:6px 0">'
                            f'FID/KID/IS are auto-computed from Generated Latent Bank per input image.<br/>'
                            f'Manual values are fallback only if latent-bank update fails.</div>')
                    score_split = gr.Dropdown(
                        choices=["train", "validation", "test"],
                        value="test",
                        label="Latent Bank Reference Split",
                    )
                    with gr.Row():
                        ref_fid = gr.Number(label="FID", value=-1.0, scale=1)
                        ref_kid = gr.Number(label="KID", value=-1.0, scale=1)
                        ref_is  = gr.Number(label="IS",  value=-1.0, scale=1)
                    load_metrics_btn = gr.Button("↻ Load from CSV", size="sm")
                score_btn = gr.Button(">> SCORE IMAGE",
                                      variant="primary", **bkw())

            with gr.Column(scale=1, min_width=270):
                gauge_out = gr.Image(label="Quality Gauge",
                                     height=210, **ikw())
                radar_out = gr.Image(label="Metric Profile",
                                     height=280, **ikw())

            with gr.Column(scale=1, min_width=250):
                bars_out    = gr.Image(label="Breakdown",
                                       height=175, **ikw())
                verdict_out = gr.HTML()
                analysis_out = gr.HTML()

        json_out = gr.JSON(label="Raw Result")

        def run_score(img, prompt, split, fid, kid, is_):
            def err(msg):
                return (None, None, None,
                    f'<div style="color:{C_RED};font-family:monospace;'
                    f'font-size:12px">&gt; {msg}</div>', "", {})
            if img is None:        return err("Upload an image first")
            if not prompt.strip(): return err("Enter a prompt first")
            if not engine._model.loaded:
                return err("No model loaded -- run pipeline_v6.py")
            try:
                r = engine.score(prompt.strip(), img,
                                 ref_fid=fid, ref_kid=kid, ref_is=is_,
                                 output_dir=output_dir,
                                 latent_split=split,
                                 use_latent_bank=True)
                display_r = clean_result_for_display(r)
                return (chart_gauge(r["composite_score"], r["verdict"]),
                        chart_radar(r), chart_bars(r),
                        verdict_html(r["verdict"], r["composite_score"],
                                     r["confidence"]),
                        analysis_html(r), display_r)
            except Exception as e:
                log.exception("score error")
                return err(str(e))

        def load_metrics_fn(split):
            fid_val, kid_val, is_val = load_metrics_from_csv(output_dir, split=split)
            return fid_val, kid_val, is_val

        load_metrics_btn.click(load_metrics_fn,
            inputs=[score_split],
            outputs=[ref_fid, ref_kid, ref_is])

        score_btn.click(run_score,
            inputs=[image_in, prompt_in, score_split, ref_fid, ref_kid, ref_is],
            outputs=[gauge_out, radar_out, bars_out,
                     verdict_out, analysis_out, json_out])

        gr.HTML(
            f'<div style="text-align:center;padding:14px;'
            f'color:{C_DIM};font-family:monospace;font-size:10px;'
            f'border-top:1px solid {C_BORDER};margin-top:6px;'
            f'text-transform:uppercase;letter-spacing:.08em">'
            f'T2I Evaluator v5.0 &nbsp;&bull;&nbsp; '
            f'train(42k)=fit &nbsp;&bull;&nbsp; '
            f'val(9k)=earlystop &nbsp;&bull;&nbsp; '
            f'test(45k)=eval only</div>')

    return demo


# ==============================================================================
#  ENTRY POINT
# ==============================================================================
def parse_args():
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser("T2I Evaluator Dashboard v5.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir",    default=str(script_dir / "_oupipeline_v5tputs/models"))
    p.add_argument("--model-file",   default=None,
                   help="Direct path to model file (e.g., evaluator_v1.pt)")
    p.add_argument("--output-dir",   default=str(script_dir / "_oupipeline_v5tputs"),
                   help="_oupipeline_v5tputs directory for Results browser")
    p.add_argument("--v4-fallback",  default=str(script_dir / "_oupipeline_v5tputs/models"))
    p.add_argument("--port",  type=int, default=7860)
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not GRADIO_OK: print("pip install gradio"); sys.exit(1)
    if not TORCH_OK:  print("pip install torch");  sys.exit(1)

    device = torch.device(
        "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda")
    log.info(f"Device: {device}")

    engine    = InferenceEngine(device)
    model_dir = args.model_dir
    model_file = args.model_file

    if model_file and Path(model_file).exists():
        try:
            engine.load_model(model_file)
            log.info(f"Model loaded from: {model_file}")
        except Exception as e:
            log.error(f"Failed to load model from {model_file}: {e}")
    else:
        mf_path   = Path(model_dir) / "manifest.json"
        if not mf_path.exists():
            fb = Path(args.v4_fallback) / "manifest.json"
            if fb.exists():
                model_dir = args.v4_fallback; mf_path = fb
                log.info(f"Using v4 fallback: {model_dir}")
            else:
                log.warning(f"No manifest.json in {model_dir}. "
                            f"Run pipeline_v6.py first.")
        if mf_path.exists():
            mf = json.loads(mf_path.read_text())
            if mf.get("latest"):
                pt = str(Path(model_dir) / mf["latest"])
                if Path(pt).exists():
                    try:
                        engine.load_model(pt)
                        log.info(f"Model loaded from manifest: {mf['latest']}")
                    except Exception as e:
                        log.error(f"Failed to load model {pt}: {e}")
                else:
                    log.warning(f"Model file not found: {pt}")

    print(f"\n{'='*60}")
    print(f"  T2I Evaluator Dashboard v5.0")
    print(f"  URL     : http://localhost:{args.port}")
    print(f"  Device  : {device}")
    print(f"  Models  : {model_dir}")
    if engine._model.loaded:
        print(f"  Loaded  : {engine._model.path}")
    else:
        print(f"  Loaded  : None (dashboard in demo mode)")
    print(f"{'='*60}\n")

    launch_port = args.port
    try:
        launch_port = find_free_port(args.port, host="127.0.0.1", max_tries=25)
        if launch_port != args.port:
            log.warning(f"Port {args.port} is busy; using {launch_port} instead")
    except OSError as e:
        log.error(str(e))
        sys.exit(1)

    kw: Dict = {}
    if GR_SHOW_API: kw["show_api"] = False
    build_dashboard(engine, model_dir, output_dir=args.output_dir).launch(
        server_name=args.host, server_port=launch_port,
        share=args.share, quiet=False, **kw)


if __name__ == "__main__":
    main()