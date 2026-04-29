"""
T2I EVALUATION PIPELINE v5.0
"A Unified Multimodal Framework for Quantitative and Semantic Evaluation
 of Text-to-Image Generative Models"
"""

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

import gc, time, json, logging, threading, contextlib, warnings, re
import platform, argparse, random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

class TqdmLoggingHandler(logging.Handler):
    """Routes logging through tqdm.write to prevent progress bar corruption."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

try:
    import psutil; PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torchvision import transforms, models
    TORCH_OK = True
    torch.backends.cudnn.benchmark = True
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

def _blip_itm_forward(model, proc, images, texts, device):
    import torch, numpy as np
    if not images or not texts:
        return None
    imgs = [im.convert("RGB") if im.mode != "RGB" else im for im in images]
    try:
        inp = proc(images=imgs, text=list(texts),
                   return_tensors="pt", padding=True,
                   truncation=True, max_length=64)
    except Exception as e:
        log.warning(f"[BLIP] Processor error: {e}")
        return None
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
        try:
            out = model(**inp, use_itm_head=True)
            if hasattr(out, "itm_score"):
                scores = torch.softmax(out.itm_score.float(), dim=-1)[:, 1]
                return scores.cpu().numpy().astype(np.float32)
            if hasattr(out, "logits"):
                lg = out.logits.float()
                if lg.dim() == 2 and lg.shape[1] == 2:
                    return torch.softmax(lg, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
                if lg.dim() == 1 or (lg.dim() == 2 and lg.shape[1] == 1):
                    return torch.sigmoid(lg.squeeze(-1)).cpu().numpy().astype(np.float32)
        except TypeError:
            pass
        except Exception as e:
            log.warning(f"[BLIP] use_itm_head=True forward failed: {e}")
        try:
            out = model(**inp)
            for attr in ("itm_score", "logits", "image_text_matching_score"):
                if not hasattr(out, attr):
                    continue
                lg = getattr(out, attr).float()
                if lg.dim() == 2 and lg.shape[1] == 2:
                    return torch.softmax(lg, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
                if lg.dim() <= 2:
                    return torch.sigmoid(lg.squeeze(-1)).cpu().numpy().astype(np.float32)
        except Exception as e:
            log.warning(f"[BLIP] plain forward failed: {e}")
        try:
            out = model(**inp, use_itm_head=False)
            for attr in ("itm_score", "logits"):
                if hasattr(out, attr):
                    lg = getattr(out, attr).float()
                    return torch.sigmoid(lg.squeeze(-1)).cpu().numpy().astype(np.float32)
        except Exception as e:
            log.warning(f"[BLIP] use_itm_head=False forward failed: {e}")
    log.warning("[BLIP] All forward strategies failed -- skipping batch")
    return None

try:
    from datasets import load_dataset; HF_DATA_OK = True
except ImportError:
    HF_DATA_OK = False

try:
    from scipy import linalg; SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLT_OK = True
except ImportError:
    PLT_OK = False

try:
    import spacy
    _SPACY_MODEL = "en_core_web_sm"
    _spacy_nlp   = None
    SPACY_OK     = True
except ImportError:
    SPACY_OK     = False

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_OK = True
except ImportError:
    NLTK_OK = False

warnings.filterwarnings("ignore")

DATASET          = "Rajarshi-Roy-research/Defactify_Image_Dataset"
ALL_SPLITS       = ["train", "validation", "test"]
SPLIT_SIZES      = {"train": 42000, "validation": 9000, "test": 45000}
LABEL_A_MAP      = {0: "Real", 1: "AI-Generated"}
LABEL_B_MAP      = {0: "Real", 1: "SD21", 2: "SDXL", 3: "SD3",
                    4: "DALLE3", 5: "Midjourney"}
LABEL_B_INV      = {v: k for k, v in LABEL_B_MAP.items()}
AI_MODELS        = ["SD21", "SDXL", "SD3", "DALLE3", "Midjourney"]
CLIP_MEAN        = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD         = (0.26862954, 0.26130258, 0.27577711)
PALETTE          = {"Real": "#34d399", "SD21": "#4f9cf9", "SDXL": "#818cf8",
                    "SD3": "#f472b6", "DALLE3": "#fb923c", "Midjourney": "#a78bfa"}
FID_MIN_SAMPLES  = 512
FID_IMAGINARY_THRESH = 1e-3
_CUDA_LOCK       = threading.Lock()

def setup_logging(output_dir: str, seed: int) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline_v4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                             datefmt="%H:%M:%S")
    fh = logging.FileHandler(Path(output_dir) / "pipeline_v4.log",
                              mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = TqdmLoggingHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    logger.info("=" * 60)
    logger.info(f"=== NEW RUN  seed={seed}  {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info("=" * 60)
    return logger

log: logging.Logger = logging.getLogger("pipeline_v4")

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    log.info(f"Global seed = {seed}")

@dataclass
class HWSnap:
    cpu_name: str = ""; cpu_phys: int = 1; cpu_logi: int = 1
    cpu_pct: float = 0; threads: int = 1
    ram_total: float = 0; ram_used: float = 0
    ram_avail: float = 0; ram_pct: float = 0
    disk_total: float = 0; disk_used: float = 0; disk_pct: float = 0
    gpu_ok: bool = False; gpu_n: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_used: float = 0; gpu_total: float = 0
    gpu_pct: float = 0; cuda_ver: str = "N/A"
    device: str = "cpu"

class ResourceManager:
    def __init__(self, cap: float = 85.0, disk_path: str = ".",
                 poll_sec: float = 30.0, force_gpu: bool = True):
        self.cap  = cap
        self.disk = os.path.abspath(disk_path)
        self.poll = poll_sec
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._disk_warn_ts: float = 0.0
        self.gpu_n = torch.cuda.device_count() if TORCH_OK else 0

        if self.gpu_n == 0:
            log.warning("No CUDA GPU -- CPU mode.") if force_gpu else \
            log.info("CPU mode.")
        self._cap_cpu(); self._cap_gpu()
        log.info(f"ResourceManager | cap={cap}% | GPUs={self.gpu_n}")

    def _cap_cpu(self):
        logi = os.cpu_count() or 1
        ok   = max(1, int(logi * self.cap / 100))
        if TORCH_OK: torch.set_num_threads(ok)
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[v] = str(ok)
        log.info(f"CPU threads -> {ok}/{logi} ({self.cap:.0f}%)")

    def _cap_gpu(self):
        if not TORCH_OK or self.gpu_n == 0: return
        frac = self.cap / 100.0
        for i in range(self.gpu_n):
            try:
                torch.cuda.set_per_process_memory_fraction(frac, device=i)
                p = torch.cuda.get_device_properties(i)
                log.info(f"GPU {i} [{p.name}] VRAM capped -> "
                         f"{frac:.0%} = {frac*p.total_memory/1024**3:.2f} GB")
            except Exception as e:
                log.warning(f"GPU {i} cap failed: {e}")

    def snap(self) -> HWSnap:
        s = HWSnap()
        s.cpu_logi = os.cpu_count() or 1
        s.threads  = max(1, int(s.cpu_logi * self.cap / 100))
        if PSUTIL_OK:
            s.cpu_pct  = psutil.cpu_percent(interval=0.05)
            s.cpu_phys = psutil.cpu_count(logical=False) or 1
            s.cpu_name = (platform.processor() or "Unknown")[:40]
            vm = psutil.virtual_memory()
            s.ram_total = vm.total/1024**3; s.ram_used = vm.used/1024**3
            s.ram_avail = vm.available/1024**3; s.ram_pct = vm.percent
            du = psutil.disk_usage(self.disk)
            s.disk_total = du.total/1024**3; s.disk_used = du.used/1024**3
            s.disk_pct   = du.percent
        if TORCH_OK and self.gpu_n > 0:
            s.gpu_ok = True; s.gpu_n = self.gpu_n
            used = total = 0.0; names = []
            for i in range(self.gpu_n):
                p = torch.cuda.get_device_properties(i)
                used  += torch.cuda.memory_allocated(i)
                total += p.total_memory; names.append(p.name)
            s.gpu_names = names; s.gpu_used = used/1024**3
            s.gpu_total = total/1024**3
            s.gpu_pct   = 100.0 * used / max(total, 1)
            s.cuda_ver  = torch.version.cuda or "N/A"; s.device = "cuda"
        return s

    def enforce(self, free: bool = True) -> bool:
        s = self.snap(); bad = []
        if PSUTIL_OK:
            if s.ram_pct > self.cap:
                bad.append(f"RAM:{s.ram_pct:.1f}%")
            if s.disk_pct > 98.0:
                now = time.time()
                if now - self._disk_warn_ts > 60:
                    self._disk_warn_ts = now
                    log.warning(f"[DISK] {s.disk_pct:.1f}% full. "
                                f"Free: {s.disk_total-s.disk_used:.1f} GB.")
        if s.gpu_ok and s.gpu_pct > self.cap:
            bad.append(f"VRAM:{s.gpu_pct:.1f}%")
        if bad:
            if free:
                ram = s.ram_pct
                if ram > 95.0:   time.sleep(1.0)
                elif ram > 90.0: time.sleep(0.5)
                log.warning(f"[WARN] Over cap -- {' | '.join(bad)} -> stalling slightly to recover")
            return False
        return True

    def _avail_bytes(self, safety: float = 0.75) -> int:
        s = self.snap()
        if s.gpu_ok and s.gpu_total > 0:
            avail_gb = s.gpu_total * (self.cap/100.0) - s.gpu_used
            return max(int(avail_gb * safety * 1024**3), 64*1024**2)
        if PSUTIL_OK:
            return max(int(s.ram_avail * safety * 1024**3), 256*1024**2)
        return 2*1024**3

    def bs(self, bps: int, maxbs: int = 256) -> int:
        return max(1, min(self._avail_bytes() // max(bps, 1), maxbs))

    def clip_bs(self)  -> int:
        return self.bs(3*224*224*4, 128 if self.gpu_n > 0 else 64)

    def inc_bs(self)   -> int:
        return self.bs(3*299*299*4, 128 if self.gpu_n > 0 else 64)

    def blip_bs(self)  -> int:
        return self.bs(3*384*384*4*3, 16 if self.gpu_n > 0 else 4)

    @property
    def device(self) -> "torch.device":
        return torch.device("cuda" if self.gpu_n > 0 else "cpu")

    def start_monitor(self):
        self._stop.clear()
        def _loop():
            while not self._stop.is_set():
                self.enforce(free=False)
                self._stop.wait(self.poll)
        self._thr = threading.Thread(target=_loop, daemon=True, name="HW-Mon")
        self._thr.start()

    def stop_monitor(self):
        self._stop.set()
        if self._thr: self._thr.join(timeout=5)

    @contextlib.contextmanager
    def monitor(self, label: str = ""):
        log.info(f"[{label}] START")
        self.start_monitor(); t0 = time.time()
        try:
            yield self
        finally:
            self.stop_monitor()
            s = self.snap()
            log.info(f"[{label}] END {time.time()-t0:.1f}s  "
                     f"RAM={s.ram_pct:.1f}%  CPU={s.cpu_pct:.1f}%"
                     + (f"  VRAM={s.gpu_pct:.1f}%" if s.gpu_ok else ""))

    def print_report(self):
        s = self.snap()
        def bar(p, w=26):
            f = int(w * min(p, 100) / 100)
            return ("[WARN] " if p > self.cap else "[OK]   ") + "#"*f + "."*(w-f) + f" {p:.1f}%"
        W = 72
        print("\n+" + "="*(W-2) + "+")
        print("|" + "  HARDWARE AUTO-DETECT -- v5.0".center(W-2) + "|")
        print(f"|  PyTorch {torch.__version__ if TORCH_OK else 'N/A'}"
              f"  |  CUDA {s.cuda_ver}  |  Cap {self.cap:.0f}%".ljust(W-1) + "|")
        print("+"+"="*(W-2)+"+")
        if PSUTIL_OK:
            print(f"|  CPU  {bar(s.cpu_pct):<55}|")
            print(f"|  RAM  {s.ram_used:.1f}/{s.ram_total:.1f} GB  "
                  f"{bar(s.ram_pct):<44}|")
            print(f"|  Disk {s.disk_used:.1f}/{s.disk_total:.1f} GB  "
                  f"{bar(s.disk_pct):<43}|")
        print("+"+"="*(W-2)+"+")
        if s.gpu_ok:
            for i, n in enumerate(s.gpu_names):
                print((f"|  GPU {i}  {n}")[:W-1].ljust(W-1) + "|")
            print(f"|  VRAM {s.gpu_used:.2f}/{s.gpu_total:.2f} GB  "
                  f"{bar(s.gpu_pct):<40}|")
            print(f"|  VRAM cap -> {s.gpu_total*self.cap/100:.2f} GB".ljust(W-1) + "|")
        else:
            print(f"|  GPU: NOT DETECTED -- CPU mode".ljust(W-1) + "|")
        print("+"+"="*(W-2)+"+")
        print(f"|  Batch sizes:  CLIP={self.clip_bs():<5}"
              f" Inc={self.inc_bs():<5} BLIP={self.blip_bs():<4}".ljust(W-1) + "|")
        print("+"+"="*(W-2)+"+\n")

class ModelRegistry:
    def __init__(self, rm: ResourceManager):
        self.rm  = rm; self.dev = rm.device
        self._clip = None; self._oc_tok = None
        self._hf_proc = None; self._clip_name = ""
        self._clip_prep = transforms.Compose([
            transforms.Resize(224,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        self._inc = None
        self._inc_prep = transforms.Compose([
            transforms.Resize((299, 299),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self._blip = None; self._blip_proc = None
        self._blip_attempted = False

    def get_clip(self):
        if self._clip is not None:
            return self._clip, self._oc_tok, self._hf_proc, self._clip_name
        if OPEN_CLIP_OK:
            log.info("Loading OpenCLIP ViT-B/32 ...")
            m, _, prep = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=self.dev)
            m.eval()
            self._clip = m; self._oc_tok = open_clip.get_tokenizer("ViT-B-32")
            self._clip_prep = prep; self._clip_name = "open_clip/ViT-B-32"
        elif HF_TRANS_OK:
            hf = "openai/clip-vit-base-patch32"
            log.info(f"Loading HF CLIP {hf} ...")
            self._clip = CLIPModel.from_pretrained(hf).to(self.dev).eval()
            self._hf_proc = CLIPProcessor.from_pretrained(hf)
            self._clip_name = f"hf/{hf}"
        else:
            raise RuntimeError("pip install open_clip_torch or transformers")
        log.info(f"CLIP ready: {self._clip_name}")
        return self._clip, self._oc_tok, self._hf_proc, self._clip_name

    @property
    def clip_prep(self): return self._clip_prep

    def get_inception(self):
        if self._inc is not None: return self._inc, self._inc_prep
        log.info("Loading InceptionV3 ...")
        inc = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        inc.aux_logits = False; inc.AuxLogits = None
        inc.eval().to(self.dev)
        self._inc = inc; log.info("InceptionV3 ready.")
        return self._inc, self._inc_prep

    def get_blip(self) -> Tuple[Optional[Any], Optional[Any]]:
        if self._blip is not None: return self._blip, self._blip_proc
        if self._blip_attempted:   return None, None
        self._blip_attempted = True
        if not HF_TRANS_OK:
            log.warning("transformers not installed -- BLIP disabled.")
            return None, None
        candidates = [
            "Salesforce/blip-itm-base-coco",
            "Salesforce/blip-itm-large-coco",
        ]
        for mn in candidates:
            try:
                log.info(f"Loading BLIP-ITM: {mn} ...")
                proc  = BlipProcessor.from_pretrained(mn)
                model = BlipForImageTextRetrieval.from_pretrained(mn)
                model = model.to(self.dev).eval()
                from PIL import Image as _Im
                _img  = _Im.new("RGB", (64, 64))
                _text = "test"
                _res  = _blip_itm_forward(model, proc, [_img], [_text], self.dev)
                if _res is None:
                    log.warning(f"[BLIP] Smoke test failed for {mn} -- skipping")
                    del model, proc; continue
                log.info(f"BLIP-ITM ready: {mn}  smoke_score={_res[0]:.4f}")
                self._blip_proc = proc; self._blip = model
                return self._blip, self._blip_proc
            except Exception as e:
                log.warning(f"BLIP load/test failed [{mn}]: {e}")
        log.warning("[BLIP] All candidate models failed -- BLIP disabled.")
        return None, None

    def unload_blip(self):
        if self._blip is not None:
            self._blip.cpu(); del self._blip; self._blip = None
            self._blip_proc = None
            if TORCH_OK and self.dev.type == "cuda":
                torch.cuda.empty_cache()
            log.info("BLIP unloaded from VRAM.")

class DataPipeline:
    def __init__(self, rm: ResourceManager, split: str,
                 max_samples: Optional[int] = None,
                 cache_dir: str = "./hf_cache"):
        self.rm          = rm
        self.split       = split
        self.max_samples = max_samples
        self.cache_dir   = cache_dir
        self.captions:   List[str] = []
        self.label_a:    List[int] = []
        self.label_b:    List[int] = []
        self.sample_ids: List[int] = []
        self._hf_ds      = None
        self._loaded     = False

    def load(self) -> "DataPipeline":
        if not HF_DATA_OK: raise RuntimeError("pip install datasets")
        log.info(f"[Phase1] Scanning {self.split} metadata ...")
        self._hf_ds = load_dataset(DATASET, split=self.split,
                                   cache_dir=self.cache_dir)
        limit = self.max_samples or len(self._hf_ds)
        limit = min(limit, len(self._hf_ds))
        for idx in tqdm(range(limit), desc=f"  meta/{self.split}",
                        unit="rows", dynamic_ncols=True, position=1, leave=False):
            row = self._hf_ds[idx]
            self.captions.append(str(row["Caption"]))
            self.label_a.append(int(row["Label_A"]))
            self.label_b.append(int(row["Label_B"]))
            self.sample_ids.append(idx)
        self._loaded = True
        log.info(f"[Phase1] {self.split}: {len(self.captions)} samples "
                 "(images streamed per-batch)")
        return self

    def iter_batches(self, batch_size: int) -> Iterator[Dict]:
        n = len(self.sample_ids)
        workers = min(16, (os.cpu_count() or 4))
        
        def fetch_img(idx):
            try:
                raw = self._hf_ds[idx]["Image"]
                return raw.convert("RGB") if raw.mode != "RGB" else raw
            except Exception as e:
                log.warning(f"Image load failure at index {idx}: {e}")
                return Image.new("RGB", (224, 224))

        with ThreadPoolExecutor(max_workers=workers) as exc:
            for start in range(0, n, batch_size):
                end  = min(start + batch_size, n)
                indices = self.sample_ids[start:end]
                
                imgs = list(exc.map(fetch_img, indices))
                
                yield {
                    "images":   imgs,
                    "captions": self.captions[start:end],
                    "label_a":  self.label_a[start:end],
                    "label_b":  self.label_b[start:end],
                    "ids":      indices,
                }
                del imgs

    def __len__(self) -> int: return len(self.sample_ids)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "sample_id": sid, "caption": self.captions[i],
            "label_a": self.label_a[i],
            "label_a_name": LABEL_A_MAP.get(self.label_a[i], "?"),
            "label_b": self.label_b[i],
            "label_b_name": LABEL_B_MAP.get(self.label_b[i], "?"),
            "split": self.split,
        } for i, sid in enumerate(self.sample_ids)])

    def statistics(self) -> Dict:
        la = {v: 0 for v in LABEL_A_MAP.values()}
        lb = {v: 0 for v in LABEL_B_MAP.values()}
        for a, b in zip(self.label_a, self.label_b):
            la[LABEL_A_MAP.get(a, "?")] += 1
            lb[LABEL_B_MAP.get(b, "?")] += 1
        return dict(split=self.split, total=len(self.captions),
                    label_a=la, label_b=lb)

    def print_statistics(self):
        st = self.statistics()
        print("\n" + "-" * 56)
        print(f"  [{st['split'].upper()}] Dataset Statistics")
        print("-" * 56)
        print(f"  Total : {st['total']:,}")
        print("\n  Label A:")
        [print(f"    {n:<18} {c:>7,}") for n, c in st["label_a"].items()]
        print("\n  Label B:")
        [print(f"    {n:<18} {c:>7,}") for n, c in st["label_b"].items()]
        print("-" * 56 + "\n")

class EmbeddingEngine:
    def __init__(self, rm: ResourceManager, reg: ModelRegistry):
        self.rm  = rm; self.reg = reg; self.dev = rm.device

    @torch.no_grad()
    def _text_chunk(self, texts: List[str]) -> np.ndarray:
        clip, tok, clip_proc, _ = self.reg.get_clip()
        with torch.autocast(device_type=self.dev.type, enabled=(self.dev.type == 'cuda')):
            if tok:
                t    = tok(texts).to(self.dev)
                feat = clip.encode_text(t, normalize=True)
            else:
                enc  = clip_proc(text=texts, return_tensors="pt",
                                 padding=True, truncation=True, max_length=77)
                feat = clip.get_text_features(
                    **{k: v.to(self.dev) for k, v in enc.items()})
        return F.normalize(feat.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def _image_chunk(self, imgs: List[Image.Image]) -> np.ndarray:
        clip, tok, clip_proc, _ = self.reg.get_clip()
        rgbs = [i.convert("RGB") if i.mode != "RGB" else i for i in imgs]
        with torch.autocast(device_type=self.dev.type, enabled=(self.dev.type == 'cuda')):
            if tok:
                prep = self.reg.clip_prep
                ts   = [prep(i) for i in rgbs]
                t    = torch.stack(ts).to(self.dev)
                feat = clip.encode_image(t, normalize=True)
            else:
                enc  = clip_proc(images=rgbs, return_tensors="pt", padding=True)
                t    = enc["pixel_values"].to(self.dev)
                feat = clip.get_image_features(pixel_values=t)
        return F.normalize(feat.float(), dim=-1).cpu().numpy()

    def embed_split(self, dp: DataPipeline,
                    batch_size: int = 16) -> Dict[str, np.ndarray]:
        N   = len(dp)
        bs  = min(batch_size, self.rm.clip_bs())
        log.info(f"[Phase2/{dp.split}] Streaming {N:,} samples, batch={bs}")

        CLIP_DIM = 512; INC_POOL = 2048; INC_LOG = 1000
        bufs: Dict[str, np.ndarray] = {
            "clip_text":  np.zeros((N, CLIP_DIM),  dtype=np.float32),
            "clip_image": np.zeros((N, CLIP_DIM),  dtype=np.float32),
            "clip_sim":   np.zeros(N,               dtype=np.float32),
            "inc_pool":   np.zeros((N, INC_POOL),   dtype=np.float32),
            "inc_log":    np.zeros((N, INC_LOG),    dtype=np.float32),
            "label_a":    np.zeros(N,               dtype=np.int32),
            "label_b":    np.zeros(N,               dtype=np.int32),
            "sample_ids": np.zeros(N,               dtype=np.int64),
        }

        inc, inc_prep = self.reg.get_inception()
        pool_buf: List[np.ndarray] = []

        def _hook(m, inp, out):
            pool_buf.append(out.squeeze(-1).squeeze(-1).cpu().float().numpy())

        handle = inc.avgpool.register_forward_hook(_hook)
        ptr    = 0
        total_b = (N + bs - 1) // bs

        try:
            for b_idx, batch in enumerate(tqdm(
                    dp.iter_batches(bs), total=total_b,
                    desc=f"  embed/{dp.split}", unit="batch", dynamic_ncols=True, position=1, leave=False)):

                self.rm.enforce()
                imgs    = batch["images"]; bsz = len(imgs)
                sl      = slice(ptr, ptr + bsz)

                te = self._text_chunk(batch["captions"])
                bufs["clip_text"][sl] = te

                ie = self._image_chunk(imgs)
                bufs["clip_image"][sl] = ie
                sims = (te * ie).sum(axis=1)
                bufs["clip_sim"][sl] = np.clip(sims, -1.0, 1.0)

                inc_bs = self.rm.inc_bs()
                ip_parts: List[np.ndarray] = []
                il_parts: List[np.ndarray] = []
                for i0 in range(0, bsz, inc_bs):
                    chunk = imgs[i0:i0+inc_bs]
                    pool_buf.clear()
                    ts = [inc_prep(img.convert("RGB")
                                   if img.mode != "RGB" else img)
                          for img in chunk]
                    _lk = (_CUDA_LOCK if self.dev.type == "cuda"
                           else contextlib.nullcontext())
                    with _lk:
                        t = torch.stack(ts).to(self.dev)
                        with torch.no_grad(), torch.autocast(device_type=self.dev.type, enabled=(self.dev.type == 'cuda')):
                            logits = inc(t)
                            if isinstance(logits, tuple): logits = logits[0]
                    assert len(pool_buf) == 1, \
                        f"Hook: expected 1, got {len(pool_buf)}"
                    ip_parts.append(pool_buf[0])
                    il_parts.append(logits.cpu().float().numpy())
                    del t, logits
                bufs["inc_pool"][sl] = np.vstack(ip_parts)
                bufs["inc_log"][sl]  = np.vstack(il_parts)

                bufs["label_a"][sl]    = np.array(batch["label_a"],  dtype=np.int32)
                bufs["label_b"][sl]    = np.array(batch["label_b"],  dtype=np.int32)
                bufs["sample_ids"][sl] = np.array(batch["ids"],      dtype=np.int64)

                ptr += bsz
                del imgs, te, ie, ip_parts, il_parts

        except KeyboardInterrupt:
            log.warning(f"\n[Phase2/{dp.split}] KeyboardInterrupt detected! Halting embedding at {ptr} samples.")
            bufs["_interrupted"] = np.array([True])
        finally:
            handle.remove()

        log.info(f"[Phase2/{dp.split}] Done. {ptr:,} samples embedded.")
        return {k: v[:ptr] for k, v in bufs.items()}

@dataclass
class AlignmentScore:
    sample_id: int; caption: str; generator: str; split: str
    clip_score: float; blip_score: Optional[float]; combined: float
    label_a: int = 0; label_b: int = 0

    def to_dict(self) -> Dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}

@dataclass
class AlignmentReport:
    scores:    List[AlignmentScore]
    gen_stats: Dict[str, Dict]
    elapsed:   float
    blip_ok:   bool
    split:     str = "?"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.scores])

    def print_summary(self):
        w = 80
        print("\n" + "-"*w)
        print(f"  Phase 3 Alignment [{self.split}] "
              f"| BLIP={'on' if self.blip_ok else 'off'} "
              f"| {self.elapsed:.1f}s")
        print("-"*w)
        hdr = f"  {'Generator':<14}{'N':>6}  {'CLIP':>8}  {'StdDev':>7}  "
        hdr += f"{'BLIP':>8}  {'Combined':>9}"
        print(hdr); print("-"*w)
        for g, st in self.gen_stats.items():
            bl = (f"{st['blip_mean']:.4f}" if st['blip_mean'] is not None
                  else "   N/A ")
            mk = " <-- real" if g == "Real" else ""
            print(f"  {g:<14}{st['n']:>6}  "
                  f"{st['clip_mean']:>8.4f}  {st['clip_std']:>7.4f}  "
                  f"{bl:>8}  {st['combined_mean']:>9.4f}{mk}")
        print("-"*w + "\n")

class SemanticAligner:
    def __init__(self, rm: ResourceManager, reg: ModelRegistry,
                 enable_blip: bool = True,
                 clip_w: float = 0.6, blip_w: float = 0.4):
        self.rm = rm; self.dev = rm.device; self.reg = reg
        self.enable_blip = enable_blip
        self.clip_w = clip_w; self.blip_w = blip_w

    @torch.no_grad()
    def _blip_chunk(self, texts: List[str],
                    imgs: List[Image.Image]) -> Optional[np.ndarray]:
        blip, proc = self.reg.get_blip()
        if blip is None: return None
        result = _blip_itm_forward(blip, proc, imgs, texts, self.dev)
        if result is None:
            log.warning("[BLIP] _blip_chunk returned None -- chunk skipped")
        return result

    def score_all(self, dp: DataPipeline,
                  emb: Dict[str, np.ndarray]) -> AlignmentReport:
        t0          = time.time()
        N           = len(emb["clip_sim"])
        clip_scores = emb["clip_sim"]

        blip_scores: Optional[np.ndarray] = None
        blip_active = False
        if self.enable_blip:
            bs  = self.rm.blip_bs()
            buf: List[np.ndarray] = []
            fail_chunks = 0
            for batch in tqdm(dp.iter_batches(bs),
                               total=(N+bs-1)//bs,
                               desc=f"  BLIP/{dp.split}", dynamic_ncols=True, position=1, leave=False):
                self.rm.enforce()
                chunk = self._blip_chunk(batch["captions"], batch["images"])
                del batch["images"]
                if chunk is None:
                    fail_chunks += 1
                    buf.append(np.full(bs, np.nan, dtype=np.float32))
                    if fail_chunks >= 3:
                        log.warning("[BLIP] 3 consecutive chunk failures -- "
                                    "disabling BLIP for this split")
                        buf = []; break
                else:
                    fail_chunks = 0
                    buf.append(chunk)
            if buf:
                arr = np.concatenate(buf)[:N]
                valid = np.sum(~np.isnan(arr))
                if valid > 0:
                    blip_scores = arr; blip_active = True
                    log.info(f"[BLIP/{dp.split}] {valid}/{N} valid scores "
                             f"({100*valid/N:.1f}%)")
                else:
                    log.warning(f"[BLIP/{dp.split}] All scores NaN -- disabled")
            self.reg.unload_blip()

        combined = (self.clip_w * clip_scores + self.blip_w * blip_scores
                    if blip_active else clip_scores.copy())

        scores = [
            AlignmentScore(
                sample_id  = dp.sample_ids[i],
                caption    = dp.captions[i],
                generator  = LABEL_B_MAP.get(emb["label_b"][i], "?"),
                split      = dp.split,
                clip_score = float(clip_scores[i]),
                blip_score = float(blip_scores[i]) if blip_active else None,
                combined   = float(combined[i]),
                label_a    = emb["label_a"][i], label_b = emb["label_b"][i],
            )
            for i in range(N)
        ]

        gen_stats: Dict[str, Dict] = {}
        for g in ["Real"] + AI_MODELS:
            sub = [x for x in scores if x.generator == g]
            if not sub: continue
            cl = np.array([x.clip_score for x in sub])
            co = np.array([x.combined   for x in sub])
            bl = np.array([x.blip_score for x in sub
                           if x.blip_score is not None])
            gen_stats[g] = dict(
                n=len(sub),
                clip_mean=float(cl.mean()), clip_std=float(cl.std()),
                blip_mean=float(bl.mean()) if len(bl) else None,
                combined_mean=float(co.mean()),
                combined_std=float(co.std()),
            )
        return AlignmentReport(scores=scores, gen_stats=gen_stats,
                               elapsed=time.time()-t0,
                               blip_ok=blip_active, split=dp.split)

def _fid(real: np.ndarray, gen: np.ndarray,
         eps: float = 1e-6) -> Tuple[float, str]:
    if not SCIPY_OK: raise RuntimeError("pip install scipy")
    warn = ""
    if len(real) < FID_MIN_SAMPLES or len(gen) < FID_MIN_SAMPLES:
        warn = f"n<{FID_MIN_SAMPLES}"
    if len(real) < 4 or len(gen) < 4:
        return -1.0, "too_few"
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sr = np.cov(real, rowvar=False) + np.eye(real.shape[1]) * eps
    sg = np.cov(gen,  rowvar=False) + np.eye(gen.shape[1])  * eps
    diff = mu_r - mu_g
    cm, _ = linalg.sqrtm(sr @ sg, disp=False)
    if np.iscomplexobj(cm):
        mx = np.abs(np.diag(cm).imag).max()
        if mx > FID_IMAGINARY_THRESH:
            warn += f" imag={mx:.4f}"
            log.critical(f"FID instability: imag={mx:.4f}")
        cm = cm.real
    raw = float(diff @ diff + np.trace(sr + sg - 2 * cm))
    if raw < 0: warn += " clamped"
    return max(raw, 0.0), warn

def _kid(real: np.ndarray, gen: np.ndarray,
         subsets: int = 100, ss: int = 1000,
         deg: int = 3) -> Tuple[float, float]:
    n_r, n_g = real.shape[0], gen.shape[0]
    ss = min(ss, n_r, n_g)
    if ss < 10: return -1.0, -1.0
    g = 1.0 / real.shape[1]
    def pk(x, y): return (g * (x @ y.T) + 1.0) ** deg
    rng = np.random.default_rng(42); vals = []
    for _ in range(subsets):
        ri = rng.choice(n_r, ss, replace=False)
        gi = rng.choice(n_g, ss, replace=False)
        r  = real[ri].astype(np.float64); ge = gen[gi].astype(np.float64)
        krr = pk(r, r); kgg = pk(ge, ge); krg = pk(r, ge)
        vals.append(float(
            (krr.sum()-np.trace(krr))/(ss*(ss-1)) +
            (kgg.sum()-np.trace(kgg))/(ss*(ss-1)) -
            2*krg.mean()))
    a = np.array(vals)
    return float(max(a.mean(), 0.0)), float(a.std())

def _is(logits: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    N = logits.shape[0]
    if N < splits * 2: return -1.0, -1.0
    def sm(x):
        e = np.exp(x - x.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    pyx = sm(logits); ss = N // splits; sc = []
    for k in range(splits):
        end = N if k == splits-1 else (k+1)*ss
        p   = pyx[k*ss:end]
        if len(p) == 0: continue
        py  = p.mean(0, keepdims=True)
        sc.append(float(np.exp(
            (p*(np.log(p+1e-10)-np.log(py+1e-10))).sum(1).mean())))
    if not sc: return -1.0, -1.0
    a = np.array(sc)
    return float(a.mean()), float(a.std())

@dataclass
class FidelityScore:
    generator: str; n_real: int; n_gen: int; split: str
    fid: float; fid_warn: str
    kid_mean: float; kid_std: float
    is_mean: float; is_std: float

    def to_dict(self) -> Dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}

@dataclass
class FidelityReport:
    scores: List[FidelityScore]; elapsed: float; split: str = "?"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.scores])

    def print_summary(self):
        if not self.scores:
            print(f"[Phase4/{self.split}] No scores."); return
        df = self.to_dataframe().sort_values("fid")
        print("\n" + "-"*80)
        print(f"  Phase 4 Fidelity [{self.split}] | {self.elapsed:.1f}s")
        print("-"*80)
        print(f"  {'Gen':<13} {'N':>6}  {'FID':>9}  "
              f"{'KID':>9}  {'IS':>7}  Warn")
        print("-"*80)
        for _, row in df.iterrows():
            km = f"{row['kid_mean']:.4f}" if row['kid_mean'] >= 0 else " N/A  "
            im = f"{row['is_mean']:.3f}"  if row['is_mean']  > 0  else " N/A "
            print(f"  {row['generator']:<13} "
                  f"{int(row['n_gen']):>6}  "
                  f"{row['fid']:>9.2f}  {km:>9}  "
                  f"{im:>7}  {str(row['fid_warn'])[:20]}")
        best = df.iloc[0]; worst = df.iloc[-1]
        print("-"*80)
        print(f"  BEST:  {best['generator']}  FID={best['fid']:.2f}")
        print(f"  WORST: {worst['generator']}  FID={worst['fid']:.2f}\n")

class FidelityEvaluator:
    def __init__(self, rm: ResourceManager,
                 compute_kid: bool = True, compute_is: bool = True):
        self.rm = rm; self.ck = compute_kid; self.ci = compute_is

    def evaluate(self, emb: Dict[str, np.ndarray],
                 split: str = "?") -> FidelityReport:
        t0       = time.time(); scores = []
        real_idx = np.where(emb["label_a"] == 0)[0]
        if len(real_idx) == 0:
            log.warning(f"[Phase4/{split}] No real images -- FID skipped.")
            return FidelityReport(scores=[], elapsed=0.0, split=split)
        real_pool = emb["inc_pool"][real_idx]
        n_real    = len(real_idx)
        
        for gen in tqdm(AI_MODELS, desc=f"  Fidelity/{split}", dynamic_ncols=True, position=1, leave=False):
            gid  = LABEL_B_INV[gen]
            gidx = np.where(emb["label_b"] == gid)[0]
            if len(gidx) == 0:
                log.warning(f"[Phase4/{split}] {gen}: no samples."); continue
            gpool = emb["inc_pool"][gidx]; glog = emb["inc_log"][gidx]
            fid_v, fw = _fid(real_pool, gpool)
            km, ks    = _kid(real_pool, gpool) if self.ck else (-1., -1.)
            im, is_   = _is(glog)              if self.ci else (-1., -1.)
            log.info(f"[Phase4/{split}] {gen:<12} "
                     f"FID={fid_v:.2f}  KID={km:.4f}  IS={im:.3f}")
            scores.append(FidelityScore(
                generator=gen, n_real=n_real, n_gen=len(gidx),
                split=split, fid=fid_v, fid_warn=fw,
                kid_mean=km, kid_std=ks, is_mean=im, is_std=is_))
        return FidelityReport(scores=scores,
                              elapsed=time.time()-t0, split=split)

class EvaluatorHead(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(0.25),
            nn.Linear(hidden, hidden//2), nn.LayerNorm(hidden//2),
            nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden//2, hidden//4), nn.GELU(),
            nn.Linear(hidden//4, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class Normalizer:
    ALL_KEYS = ["clip_sim", "blip_score", "neg_fid", "neg_kid", "is_score",
                 "robustness_score"]

    def __init__(self, active_keys: Optional[List[str]] = None):
        self.active_keys = active_keys or self.ALL_KEYS
        self.stats: Dict[str, Tuple[float, float]] = {}

    def fit(self, d: Dict[str, np.ndarray]):
        for k in self.active_keys:
            arr   = d[k].astype(np.float32)
            valid = arr[~np.isnan(arr) & (arr > -990)]
            mu    = float(valid.mean()) if len(valid) else 0.0
            std   = float(valid.std())  if len(valid) else 1.0
            self.stats[k] = (mu, max(std, 1e-8))

    def transform(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for k in self.active_keys:
            mu, std = self.stats.get(k, (0.0, 1.0))
            arr = d.get(k, np.array([np.nan])).astype(np.float32)
            arr = np.where(np.isnan(arr) | (arr < -990),
                           0.0, (arr - mu) / std)
            cols.append(arr)
        return np.stack(cols, axis=1)

    def to_dict(self) -> Dict:
        return {"active_keys": self.active_keys, "stats": self.stats}

    def from_dict(self, d: Dict):
        self.active_keys = d.get("active_keys", self.ALL_KEYS)
        self.stats = {k: tuple(v) for k, v in d.get("stats", {}).items()}

class EvaluatorModel:
    VERSION = "5.0"

    def __init__(self, device: "torch.device",
                 active_keys: Optional[List[str]] = None):
        self.device      = device
        self.active_keys = active_keys or Normalizer.ALL_KEYS
        self.norm        = Normalizer(self.active_keys)
        self.head        = EvaluatorHead(
            in_dim=len(self.active_keys)).to(device)
        self.trained     = False
        self.meta:       Dict = {}
        self.threshold:  float = 0.5
        self.train_history: List[Dict] = []

    def _prep(self, clip_sim, blip_scores, fid_arr, kid_arr, is_arr,
              rob_arr=None) -> Dict[str, np.ndarray]:
        def safe_neg(a):
            out = -a.astype(np.float32).copy()
            out[a < -990] = np.nan; return out
        def safe_arr(a):
            if a is None: return np.full(len(clip_sim), np.nan, dtype=np.float32)
            return a.astype(np.float32)
        blip = blip_scores.astype(np.float32).copy()
        blip[blip < -990] = np.nan
        d = dict(
            clip_sim        = clip_sim.astype(np.float32),
            blip_score      = blip,
            neg_fid         = safe_neg(fid_arr),
            neg_kid         = safe_neg(kid_arr),
            is_score        = is_arr.astype(np.float32),
            robustness_score= safe_arr(rob_arr),
        )
        return {k: d[k] for k in self.active_keys if k in d}

    def train(self, clip_sim, blip_scores, fid_arr, kid_arr, is_arr,
              label_a: np.ndarray,
              rob_arr=None,
              epochs: int = 150, lr: float = 5e-4,
              patience: int = 15):
        
        d_train = self._prep(clip_sim, blip_scores, fid_arr, kid_arr, is_arr,
                             rob_arr)
        
        self.norm.fit(d_train)
        X_train_full = self.norm.transform(d_train)
        y_train_full = (1.0 - label_a.astype(np.float32))
        N_train_full = len(y_train_full)

        rng   = np.random.default_rng(42)
        idx   = rng.permutation(N_train_full)
        n_val = max(int(N_train_full * 0.15), 1)
        v_idx = idx[:n_val]
        t_idx = idx[n_val:]
        
        X_val   = X_train_full[v_idx]
        y_val   = y_train_full[v_idx]
        X_train = X_train_full[t_idx]
        y_train = y_train_full[t_idx]
        N_val   = len(y_val)
        N_train = len(y_train)

        Xt = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        Xv = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        yv = torch.tensor(y_val,   dtype=torch.float32).to(self.device)

        opt   = torch.optim.AdamW(self.head.parameters(),
                                   lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, steps_per_epoch=1, epochs=epochs)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val = float("inf"); best_state = None; stale = 0
        log.info(f"[EvalModel] Full-dataset Train={N_train:,} | Internal Val={N_val:,}")

        for ep in tqdm(range(epochs), desc="  Train/Epochs", dynamic_ncols=True, position=1, leave=False):
            self.head.train()
            opt.zero_grad()
            
            # NOTE: Removed autocast context wrapper here. 
            # Torch explicitly bans binary_cross_entropy inside fp16 domains.
            pred = self.head(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            opt.step(); sched.step()
            self.head.eval()
            with torch.no_grad():
                vl = loss_fn(self.head(Xv), yv).item()
            self.train_history.append(
                {"epoch": ep+1, "train_loss": loss.item(), "val_loss": vl})
            if (ep+1) % 20 == 0:
                log.info(f"  ep {ep+1:3d}/{epochs}  "
                         f"train={loss.item():.5f}  val={vl:.5f}  "
                         f"lr={sched.get_last_lr()[0]:.2e}")
            if vl < best_val:
                best_val = vl; stale = 0
                best_state = {k: v.clone()
                              for k, v in self.head.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state: self.head.load_state_dict(best_state)
        self.head.eval(); self.trained = True

        with torch.no_grad():
            pv = torch.sigmoid(self.head(Xv)).cpu().numpy()
        y_qual = yv.cpu().numpy()
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_qual, pv)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            thr_youden = float(thresholds[best_idx])
            if 0.02 <= thr_youden <= 0.98:
                self.threshold = thr_youden
            else:
                self.threshold = float(np.median(pv))
        except Exception as thr_e:
            self.threshold = float(np.median(pv))

    @torch.no_grad()
    def predict(self, clip_sim: float, blip_score: Optional[float],
                fid: float = -1.0, kid: float = -1.0,
                is_score: float = -1.0) -> Dict:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        bs  = float(blip_score) if blip_score is not None else np.nan
        ks  = float(kid)        if kid  >= 0             else np.nan
        is_ = float(is_score)   if is_score > 0          else np.nan
        d   = dict(
            clip_sim   = np.array([clip_sim]),
            blip_score = np.array([bs]),
            neg_fid    = np.array([-fid]),
            neg_kid    = np.array([-ks] if not np.isnan(ks) else [np.nan]),
            is_score   = np.array([is_]),
        )
        d = {k: d[k] for k in self.active_keys if k in d}
        x = self.norm.transform(d)
        logits = self.head(torch.tensor(x, dtype=torch.float32).to(self.device))
        p = torch.sigmoid(logits).item()
        confidence = abs(p - 0.5) * 2.0
        verdict    = ("High Quality" if p > 0.70 else
                      "Medium Quality" if p > 0.45 else "Low Quality")
        return {
            "composite_score": round(p, 4),
            "confidence":      round(confidence, 4),
            "verdict":         verdict,
            "explanation":     self._explain(clip_sim, bs, fid, ks, is_),
            "clip_score":      round(clip_sim, 4),
            "blip_score":      round(bs, 4) if not np.isnan(bs) else None,
            "fid":             round(fid, 4),
            "kid":             round(ks, 4) if not np.isnan(ks) else None,
            "is_score":        round(is_, 4) if not np.isnan(is_) else None,
            "threshold":       round(self.threshold, 4),
        }

    def _explain(self, clip: float, blip: float, fid: float,
                 kid: float, is_: float) -> str:
        parts = []
        if   clip > 0.25: parts.append(f"Strong text-image alignment (CLIP={clip:.3f})")
        elif clip > 0.15: parts.append(f"Moderate alignment (CLIP={clip:.3f})")
        else:             parts.append(f"Weak alignment (CLIP={clip:.3f})")
        if not np.isnan(blip):
            lbl = "good" if blip > 0.5 else "poor"
            parts.append(f"BLIP match={blip:.3f} ({lbl})")
        if fid >= 0:
            lbl = "close to real" if fid < 30 else ("moderate" if fid < 100
                  else "far from real")
            parts.append(f"FID={fid:.1f} ({lbl})")
        if not np.isnan(is_) and is_ > 0:
            parts.append(f"IS={is_:.2f}")
        return " | ".join(parts) if parts else "Insufficient metrics."

    def save(self, path: str):
        torch.save({
            "version":       self.VERSION,
            "state_dict":    self.head.state_dict(),
            "normalizer":    self.norm.to_dict(),
            "meta":          self.meta,
            "threshold":     self.threshold,
            "active_keys":   self.active_keys,
            "in_dim":        len(self.active_keys),
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_history": self.train_history,
        }, path)
        log.info(f"[EvalModel] Saved -> {path}")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=True)
        in_dim = ck.get("in_dim", 5)
        self.head = EvaluatorHead(in_dim=in_dim).to(self.device)
        self.head.load_state_dict(ck["state_dict"]); self.head.eval()
        self.norm.from_dict(ck["normalizer"])
        self.meta          = ck.get("meta", {})
        self.threshold     = ck.get("threshold", 0.5)
        self.active_keys   = ck.get("active_keys", Normalizer.ALL_KEYS)
        self.train_history = ck.get("train_history", [])
        self.trained       = True

class ModelManager:
    def __init__(self, model_dir: str):
        self.dir = Path(model_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.mf  = self.dir / "manifest.json"

    def _load_mf(self) -> Dict:
        return (json.loads(self.mf.read_text()) if self.mf.exists()
                else {"versions": [], "latest": None})

    def _save_mf(self, m: Dict):
        self.mf.write_text(json.dumps(m, indent=2))

    def save(self, ev: EvaluatorModel, meta: Dict = {}) -> str:
        m = self._load_mf(); v = len(m["versions"]) + 1
        name = f"evaluator_v{v}.pt"; path = str(self.dir / name)
        ev.meta = {**meta, "version": v}; ev.save(path)
        m["versions"].append({
            "version": v, "file": name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": ev.threshold, "active_keys": ev.active_keys,
            **meta,
        })
        m["latest"] = name; self._save_mf(m)
        (self.dir / f"summary_v{v}.txt").write_text(
            f"T2I Evaluator v{v} (pipeline v5.0)\n"
            f"Saved    : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Threshold: {ev.threshold:.4f}\n"
            f"Features : {ev.active_keys}\n"
            f"Meta     : {json.dumps(meta, indent=2)}\n")
        log.info(f"ModelManager: saved v{v} -> {path}"); return path

    def load_latest(self, ev: EvaluatorModel) -> str:
        m = self._load_mf()
        if not m["latest"]: raise FileNotFoundError("No saved models.")
        path = str(self.dir / m["latest"]); ev.load(path); return path

    def list_versions(self) -> pd.DataFrame:
        m = self._load_mf()
        return (pd.DataFrame(m["versions"]) if m["versions"]
                else pd.DataFrame())

def _da(ax):
    ax.set_facecolor("#12141f")
    ax.tick_params(colors="#cbd5e1")
    for sp in ax.spines.values(): sp.set_color("#2d3148")

def plot_alignment(reports: List[AlignmentReport], path: str):
    if not PLT_OK or not reports: return
    merged: Dict[str, Dict] = {}
    for r in reports:
        for g, st in r.gen_stats.items():
            if g not in merged: merged[g] = {"clip": [], "blip": [], "comb": []}
            for _ in range(st["n"]):
                merged[g]["clip"].append(st["clip_mean"])
                merged[g]["blip"].append(st["blip_mean"] or 0.0)
                merged[g]["comb"].append(st["combined_mean"])
    gens = [g for g in ["Real"] + AI_MODELS if g in merged]
    x  = np.arange(len(gens)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(10, len(gens)*1.8), 5))
    fig.patch.set_facecolor("#0d0f1a"); _da(ax)
    for offset, key, label, col in [
        (-w, "clip", "CLIPScore",  "#60a5fa"),
        (0,  "blip", "BLIP-ITM",   "#c084fc"),
        (w,  "comb", "Combined",   "#34d399"),
    ]:
        vals = [float(np.mean(merged[g][key])) for g in gens]
        bars = ax.bar(x+offset, vals, w, label=label,
                      color=col, alpha=0.85, edgecolor="#1e2035")
        for bar, v in zip(bars, vals):
            if v > 0.001:
                ax.text(bar.get_x()+bar.get_width()/2, v+0.004,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(gens, color="#cbd5e1", fontsize=11)
    ax.set_ylabel("Score", color="#cbd5e1")
    ax.set_title("Semantic Alignment (all splits merged)",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e2035", edgecolor="#2d3148", labelcolor="white")
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1]*1.12))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(); log.info(f"Alignment chart -> {path}")

def plot_fidelity(reports: List[FidelityReport], path: str):
    if not PLT_OK: return
    all_rows = []
    for r in reports:
        for s in r.scores: all_rows.append(s.to_dict())
    if not all_rows: return
    df = pd.DataFrame(all_rows)
    agg = df.groupby("generator")[["fid","kid_mean","is_mean"]].mean()
    agg = agg.sort_values("fid"); gens = agg.index.tolist()
    cols = [PALETTE.get(g, "#888") for g in gens]
    x = np.arange(len(gens))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#0d0f1a")
    for ax, (col, title, hi) in zip(axes, [
        ("fid",      "FID (lower = better)",  False),
        ("kid_mean", "KID (lower = better)",  False),
        ("is_mean",  "IS  (higher = better)", True),
    ]):
        _da(ax)
        vals = agg[col].clip(lower=0).values
        bars = ax.bar(x, vals, color=cols, alpha=0.85, edgecolor="#1e2035")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        v+vals.max()*0.025, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color="white")
        best = int(vals.argmin()) if not hi else int(vals.argmax())
        bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(gens, color="#cbd5e1", fontsize=10, rotation=15)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    splits_str = "+".join(sorted({r.split for r in reports}))
    fig.suptitle(f"Visual Fidelity [{splits_str}]",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(); log.info(f"Fidelity chart -> {path}")

def plot_training_curve(history: List[Dict], path: str):
    if not PLT_OK or not history: return
    eps   = [h["epoch"]      for h in history]
    tl    = [h["train_loss"] for h in history]
    vl    = [h["val_loss"]   for h in history]
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0d0f1a"); _da(ax)
    ax.plot(eps, tl, color="#60a5fa", linewidth=1.8, label="Train Loss")
    ax.plot(eps, vl, color="#f472b6", linewidth=1.8, label="Val Loss")
    best_ep = eps[int(np.argmin(vl))]
    ax.axvline(best_ep, color="#fbbf24", linewidth=1.2,
               linestyle="--", label=f"Best ep={best_ep}")
    ax.set_xlabel("Epoch", color="#cbd5e1")
    ax.set_ylabel("BCE Loss", color="#cbd5e1")
    ax.set_title("Evaluator Model Training Curve",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e2035", edgecolor="#2d3148", labelcolor="white")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(); log.info(f"Training curve -> {path}")

def plot_split_comparison(al_reports: List[AlignmentReport],
                          fi_reports: List[FidelityReport], path: str):
    if not PLT_OK: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0f1a")
    SPLIT_COL = {"train": "#60a5fa", "validation": "#34d399", "test": "#f472b6"}
    ax = axes[0]; _da(ax)
    gens  = AI_MODELS
    x     = np.arange(len(gens)); w = 0.25
    for i, r in enumerate(al_reports):
        if not r: continue
        offset = (i - len(al_reports)/2 + 0.5) * w
        vals   = [r.gen_stats.get(g, {}).get("clip_mean", 0) for g in gens]
        ax.bar(x+offset, vals, w,
               label=r.split,
               color=SPLIT_COL.get(r.split, "#888"),
               alpha=0.8, edgecolor="#1e2035")
    ax.set_xticks(x)
    ax.set_xticklabels(gens, color="#cbd5e1", fontsize=10)
    ax.set_title("CLIPScore by Split", color="white", fontweight="bold")
    ax.legend(facecolor="#1e2035", edgecolor="#2d3148", labelcolor="white")
    ax = axes[1]; _da(ax)
    for i, r in enumerate(fi_reports):
        if not r: continue
        df = r.to_dataframe()
        if df.empty: continue
        offset = (i - len(fi_reports)/2 + 0.5) * w
        vals   = [df[df.generator==g]["fid"].values[0]
                  if g in df.generator.values else 0 for g in gens]
        ax.bar(x+offset, vals, w,
               label=r.split,
               color=SPLIT_COL.get(r.split, "#888"),
               alpha=0.8, edgecolor="#1e2035")
    ax.set_xticks(x)
    ax.set_xticklabels(gens, color="#cbd5e1", fontsize=10)
    ax.set_title("FID by Split", color="white", fontweight="bold")
    ax.legend(facecolor="#1e2035", edgecolor="#2d3148", labelcolor="white")
    fig.suptitle("Cross-Split Comparison", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(); log.info(f"Split comparison -> {path}")

def plot_ranking(al_reports: List[AlignmentReport],
                 fi_reports: List[FidelityReport], path: str):
    if not PLT_OK: return
    clip_agg: Dict[str, List[float]] = {}
    blip_agg: Dict[str, List[float]] = {}
    for r in al_reports:
        if not r: continue
        for g, st in r.gen_stats.items():
            if g == "Real": continue
            clip_agg.setdefault(g, []).append(st["clip_mean"])
            if st["blip_mean"] is not None:
                blip_agg.setdefault(g, []).append(st["blip_mean"])
    fid_agg: Dict[str, List[float]] = {}
    is_agg:  Dict[str, List[float]] = {}
    for r in fi_reports:
        if not r: continue
        for s in r.scores:
            fid_agg.setdefault(s.generator, []).append(s.fid)
            if s.is_mean > 0:
                is_agg.setdefault(s.generator, []).append(s.is_mean)
    rows = []
    for g in AI_MODELS:
        rows.append({
            "Generator": g,
            "CLIP":   f"{np.mean(clip_agg.get(g,[0])):.4f}",
            "BLIP":   (f"{np.mean(blip_agg[g]):.4f}"
                       if g in blip_agg else "N/A"),
            "FID":    f"{np.mean(fid_agg.get(g,[-1])):.2f}",
            "IS":     (f"{np.mean(is_agg[g]):.3f}"
                       if g in is_agg else "N/A"),
        })
    if not rows: return
    df   = pd.DataFrame(rows)
    cols = list(df.columns)
    fig, ax = plt.subplots(figsize=(len(cols)*1.9, len(rows)*0.7+2))
    fig.patch.set_facecolor("#0d0f1a"); ax.set_facecolor("#0d0f1a")
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=cols,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            g = df.iloc[r-1]["Generator"]
            cell.set_facecolor("#1a1d2e")
            cell.set_text_props(
                color=PALETTE.get(g,"white") if c==0 else "#e2e8f0")
        cell.set_edgecolor("#2d3148")
    ax.set_title("Final Generator Ranking (avg across all splits)",
                 color="white", fontsize=13, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(); log.info(f"Ranking -> {path}")

def _get_nlp():
    global _spacy_nlp
    if not SPACY_OK:
        return None
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load(_SPACY_MODEL)
        except OSError:
            log.warning(f"[spaCy] Model '{_SPACY_MODEL}' not found. "
                        "Run: python -m spacy download en_core_web_sm")
            return None
    return _spacy_nlp

@dataclass
class SemanticComponents:
    caption:    str
    objects:    List[str]
    attributes: List[str]
    relations:  List[str]
    raw_tokens: List[str]

    def to_dict(self) -> Dict:
        return {
            "caption":    self.caption,
            "objects":    "|".join(self.objects),
            "attributes": "|".join(self.attributes),
            "relations":  "|".join(self.relations),
            "n_objects":  len(self.objects),
            "n_attrs":    len(self.attributes),
            "n_rels":     len(self.relations),
        }

_REL_PREPS = {
    "on", "in", "at", "beside", "next", "near", "behind", "front",
    "above", "below", "under", "over", "between", "among", "inside",
    "outside", "around", "against", "along", "across", "through",
    "left", "right", "top", "bottom",
}

_ACT_VERBS = {
    "stand", "sit", "walk", "run", "eat", "hold", "carry", "wear",
    "look", "ride", "fly", "jump", "sleep", "play", "drive", "park",
    "lean", "hang", "climb", "swim", "lie", "float", "cross",
}

_DET_WORDS = {"a", "an", "the", "this", "that", "these", "those"}
_GENERIC_OBJECTS = {
    "photo", "picture", "image", "scene", "shot", "view",
    "portrait", "frame", "video", "drawing", "painting", "illustration",
    "thing", "object", "item", "stuff", "something",
}


def _normalize_concept_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [tok for tok in text.split() if tok not in _DET_WORDS]
    return " ".join(tokens)


def _concept_query_variants(concept: str, prefix: str) -> List[str]:
    concept = _normalize_concept_text(concept)
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

def decompose_caption(caption: str) -> SemanticComponents:
    nlp = _get_nlp()
    if nlp is None:
        words = [_normalize_concept_text(w) for w in caption.split()]
        words = [w for w in words if w and len(w) > 2 and w not in _GENERIC_OBJECTS]
        return SemanticComponents(caption=caption, objects=words,
                                  attributes=[], relations=[], raw_tokens=words)

    doc     = nlp(caption)
    objects:    List[str] = []
    attributes: List[str] = []
    relations:  List[str] = []
    raw_tokens: List[str] = []

    for chunk in doc.noun_chunks:
        phrase = _normalize_concept_text(chunk.text)
        root_lemma = chunk.root.lemma_.lower()
        if phrase and phrase not in _GENERIC_OBJECTS:
            objects.append(phrase)
        if root_lemma not in _DET_WORDS and root_lemma not in _GENERIC_OBJECTS:
            objects.append(root_lemma)
        for tok in chunk:
            if tok.pos_ == "ADJ":
                attributes.append(tok.lemma_.lower())

    for tok in doc:
        lem = tok.lemma_.lower()
        if tok.pos_ == "ADP" and lem in _REL_PREPS:
            relations.append(lem)
        if tok.pos_ == "VERB" and lem in _ACT_VERBS:
            relations.append(lem)
        if not tok.is_stop and not tok.is_punct and tok.pos_ != "DET":
            raw_tokens.append(lem)

    def _dedup(lst): return list(dict.fromkeys(lst))
    return SemanticComponents(
        caption    = caption,
        objects    = _dedup(objects)    or ["(none)"],
        attributes = _dedup(attributes) or [],
        relations  = _dedup(relations)  or [],
        raw_tokens = _dedup(raw_tokens),
    )

class SemanticDecomposer:
    def decompose_all(self, captions: List[str],
                      split: str = "?") -> List[SemanticComponents]:
        t0 = time.time()
        results = []
        for cap in tqdm(captions,
                        desc=f"  Semantic/{split}", ncols=80, unit="cap", dynamic_ncols=True, position=1, leave=False):
            results.append(decompose_caption(cap))
        elapsed = time.time() - t0
        log.info(f"[Phase6/{split}] Decomposed {len(results):,} captions "
                 f"in {elapsed:.1f}s")
        return results

    @staticmethod
    def to_dataframe(comps: List[SemanticComponents]) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in comps])

@dataclass
class FaithfulnessScore:
    sample_id:       int
    caption:         str
    generator:       str
    split:           str
    obj_score:       float
    attr_score:      float
    rel_score:       float
    faithfulness:    float
    missing_objects: List[str]
    halluc_rate:     float

    def to_dict(self) -> Dict:
        return {
            "sample_id":       self.sample_id,
            "caption":         self.caption,
            "generator":       self.generator,
            "split":           self.split,
            "obj_score":       round(self.obj_score,  4),
            "attr_score":      round(self.attr_score, 4),
            "rel_score":       round(self.rel_score,  4),
            "faithfulness":    round(self.faithfulness, 4),
            "missing_objects": "|".join(self.missing_objects),
            "halluc_rate":     round(self.halluc_rate, 4),
        }

@dataclass
class FaithfulnessReport:
    scores:  List[FaithfulnessScore]
    elapsed: float
    split:   str = "?"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.scores])

    def mean_faithfulness(self) -> float:
        if not self.scores: return 0.0
        return float(np.mean([s.faithfulness for s in self.scores]))

    def print_summary(self):
        if not self.scores:
            print(f"[Phase7/{self.split}] No scores."); return
        w = 80
        print("\n" + "-"*w)
        print(f"  Phase 7 Faithfulness [{self.split}] | {self.elapsed:.1f}s")
        print("-"*w)
        df = self.to_dataframe()
        gens = df["generator"].unique()
        print(f"  {'Generator':<14}{'N':>6}  {'Obj':>7}  "
              f"{'Attr':>7}  {'Rel':>7}  {'Faith':>8}  {'Halluc%':>8}")
        print("-"*w)
        for g in sorted(gens):
            sub = df[df.generator == g]
            print(f"  {g:<14}{len(sub):>6}  "
                  f"{sub.obj_score.mean():>7.4f}  "
                  f"{sub.attr_score.mean():>7.4f}  "
                  f"{sub.rel_score.mean():>7.4f}  "
                  f"{sub.faithfulness.mean():>8.4f}  "
                  f"{sub.halluc_rate.mean()*100:>7.1f}%")
        print("-"*w)
        print(f"  Overall faithfulness: {df.faithfulness.mean():.4f}  "
              f"Hallucination rate: {df.halluc_rate.mean()*100:.1f}%\n")

_PRESENCE_THRESH = 0.20

class FaithfulnessEvaluator:
    def __init__(self, rm: ResourceManager, reg: ModelRegistry):
        self.rm  = rm
        self.reg = reg
        self.dev = rm.device

    @torch.no_grad()
    def _clip_text_embed(self, texts: List[str]) -> np.ndarray:
        clip_model, oc_tok, hf_proc, clip_name = self.reg.get_clip()
        with torch.autocast(device_type=self.dev.type, enabled=(self.dev.type == 'cuda')):
            if oc_tok is not None:
                tokens = oc_tok(texts).to(self.dev)
                feats  = clip_model.encode_text(tokens)
            else:
                enc = hf_proc(text=texts, return_tensors="pt",
                              padding=True, truncation=True, max_length=77)
                enc = {k: v.to(self.dev) for k, v in enc.items()}
                feats = clip_model.get_text_features(**enc)
            feats  = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def _presence(self, img_emb: np.ndarray, concepts: List[str],
                  prefix: str = "a photo of") -> float:
        if not concepts or concepts == ["(none)"]:
            return 0.5
        groups = []
        for c in concepts[:8]:
            q = _concept_query_variants(c, prefix)
            if q:
                groups.append(q)
        queries = [q for group in groups for q in group]
        if not queries:
            return 0.5
        try:
            t_embs = self._clip_text_embed(queries)
            img    = img_emb / (np.linalg.norm(img_emb) + 1e-8)
            sims   = (t_embs @ img).clip(-1.0, 1.0)
            idx = 0
            concept_scores = []
            for group in groups:
                g = sims[idx:idx + len(group)]
                if len(g):
                    concept_scores.append(float(np.max(g)))
                idx += len(group)
            return float(np.mean(concept_scores)) if concept_scores else 0.5
        except Exception as e:
            log.warning(f"[Phase7] presence error: {e}")
            return 0.0

    def evaluate(self, emb: Dict[str, np.ndarray],
                 comps: List[SemanticComponents],
                 dp_captions: List[str],
                 dp_label_b:  List[int],
                 dp_sample_ids: List[int],
                 split: str = "?") -> FaithfulnessReport:
        if not OPEN_CLIP_OK:
            log.warning("[Phase7] open_clip not available -- skipping.")
            return FaithfulnessReport(scores=[], elapsed=0.0, split=split)

        t0      = time.time()
        N       = len(comps)
        scores  = []
        clip_imgs = emb["clip_image"]

        for i in tqdm(range(N), desc=f"  Faithful/{split}", ncols=80, dynamic_ncols=True, position=1, leave=False):
            comp     = comps[i]
            img_emb  = clip_imgs[i]

            obj_s    = self._presence(img_emb, comp.objects,    "a photo of")
            attr_s   = self._presence(img_emb, comp.attributes, "a photo with")
            rel_s    = self._presence(img_emb, comp.relations,  "a photo showing")

            faith = 0.5 * obj_s + 0.3 * attr_s + 0.2 * rel_s

            missing = []
            if comp.objects != ["(none)"]:
                for obj in comp.objects:
                    s = self._presence(img_emb, [obj], "a photo of")
                    if s < _PRESENCE_THRESH:
                        missing.append(obj)
            halluc_rate = len(missing) / max(len(comp.objects), 1)

            gen = LABEL_B_MAP.get(dp_label_b[i], "?")
            scores.append(FaithfulnessScore(
                sample_id       = dp_sample_ids[i],
                caption         = comp.caption,
                generator       = gen,
                split           = split,
                obj_score       = obj_s,
                attr_score      = attr_s,
                rel_score       = rel_s,
                faithfulness    = faith,
                missing_objects = missing,
                halluc_rate     = halluc_rate,
            ))

        elapsed = time.time() - t0
        log.info(f"[Phase7/{split}] Done. mean_faithfulness="
                 f"{np.mean([s.faithfulness for s in scores]):.4f}  "
                 f"mean_halluc={np.mean([s.halluc_rate for s in scores])*100:.1f}%")
        return FaithfulnessReport(scores=scores, elapsed=elapsed, split=split)

@dataclass
class RobustnessScore:
    sample_id:      int
    caption:        str
    generator:      str
    split:          str
    base_clip:      float
    perturbed_clips: List[float]
    robustness_std: float
    robustness_score: float
    perturbations:  List[str]

    def to_dict(self) -> Dict:
        return {
            "sample_id":       self.sample_id,
            "caption":         self.caption,
            "generator":       self.generator,
            "split":           self.split,
            "base_clip":       round(self.base_clip,       4),
            "robustness_std":  round(self.robustness_std,  4),
            "robustness_score":round(self.robustness_score,4),
            "n_perturbations": len(self.perturbations),
        }

@dataclass
class RobustnessReport:
    scores:  List[RobustnessScore]
    elapsed: float
    split:   str = "?"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.scores])

    def mean_robustness(self) -> float:
        if not self.scores: return 0.0
        return float(np.mean([s.robustness_score for s in self.scores]))

    def print_summary(self):
        if not self.scores:
            print(f"[Phase8/{self.split}] No scores."); return
        w = 80
        print("\n" + "-"*w)
        print(f"  Phase 8 Robustness [{self.split}] | {self.elapsed:.1f}s")
        print("-"*w)
        df = self.to_dataframe()
        gens = df["generator"].unique()
        print(f"  {'Generator':<14}{'N':>6}  "
              f"{'BaseClip':>9}  {'RobStd':>8}  {'RobScore':>9}")
        print("-"*w)
        for g in sorted(gens):
            sub = df[df.generator == g]
            print(f"  {g:<14}{len(sub):>6}  "
                  f"{sub.base_clip.mean():>9.4f}  "
                  f"{sub.robustness_std.mean():>8.4f}  "
                  f"{sub.robustness_score.mean():>9.4f}")
        print("-"*w)
        print(f"  Overall robustness: {df.robustness_score.mean():.4f}\n")

_COLOUR_SYNONYMS: Dict[str, List[str]] = {
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

_REL_SWAPS: Dict[str, str] = {
    "beside": "near", "next":  "close to", "behind": "in front of",
    "above":  "below", "under": "over",    "inside": "outside",
    "on":     "beside", "in":   "near",
}

def _perturb_caption(caption: str, comp: SemanticComponents,
                     n: int = 3, seed: int = 42) -> List[str]:
    rng   = random.Random(seed)
    perturbed: List[str] = []
    text  = caption

    for attr in rng.sample(comp.attributes, min(len(comp.attributes), 2)):
        syns = _COLOUR_SYNONYMS.get(attr.lower(), [])
        if syns:
            syn = rng.choice(syns)
            p   = text.replace(attr, syn, 1)
            if p != text and p not in perturbed:
                perturbed.append(p)
            if len(perturbed) >= n: return perturbed[:n]

    for rel in rng.sample(comp.relations, min(len(comp.relations), 2)):
        swap = _REL_SWAPS.get(rel.lower())
        if swap:
            p = text.replace(rel, swap, 1)
            if p != text and p not in perturbed:
                perturbed.append(p)
            if len(perturbed) >= n: return perturbed[:n]

    if len(perturbed) < n:
        prefixes = ["An image showing", "A picture of", "A photo depicting"]
        for pf in prefixes:
            p = f"{pf} {text[0].lower()}{text[1:]}"
            if p not in perturbed:
                perturbed.append(p)
            if len(perturbed) >= n: break

    return perturbed[:n]

class RobustnessTester:
    def __init__(self, rm: ResourceManager, reg: ModelRegistry,
                 n_perturbations: int = 3):
        self.rm  = rm
        self.reg = reg
        self.dev = rm.device
        self.n   = n_perturbations

    @torch.no_grad()
    def _clip_sim_batch(self, texts: List[str],
                        img_emb: np.ndarray) -> List[float]:
        if not OPEN_CLIP_OK and not HF_TRANS_OK:
            return []
        if not texts:
            return []
        clip_model, oc_tok, hf_proc, clip_name = self.reg.get_clip()
        with torch.autocast(device_type=self.dev.type, enabled=(self.dev.type == 'cuda')):
            if oc_tok is not None:
                tokens = oc_tok(texts).to(self.dev)
                t_embs = clip_model.encode_text(tokens)
            else:
                enc = hf_proc(text=texts, return_tensors="pt",
                              padding=True, truncation=True, max_length=77)
                enc = {k: v.to(self.dev) for k, v in enc.items()}
                t_embs = clip_model.get_text_features(**enc)
            t_embs = t_embs / t_embs.norm(dim=-1, keepdim=True)
            img    = torch.tensor(img_emb, dtype=torch.float32,
                                  device=self.dev)
            img    = img / img.norm()
            sims   = (t_embs @ img).cpu().float().numpy()
        return [float(s) for s in sims]

    def evaluate(self, emb: Dict[str, np.ndarray],
                 comps: List[SemanticComponents],
                 dp_captions:   List[str],
                 dp_label_b:    List[int],
                 dp_sample_ids: List[int],
                 split: str = "?") -> RobustnessReport:
        if not OPEN_CLIP_OK:
            log.warning("[Phase8] open_clip not available -- skipping.")
            return RobustnessReport(scores=[], elapsed=0.0, split=split)

        t0     = time.time()
        N      = len(comps)
        scores = []
        clip_imgs = emb["clip_image"]

        for i in tqdm(range(N), desc=f"  Robust/{split}", ncols=80, dynamic_ncols=True, position=1, leave=False):
            comp    = comps[i]
            img_emb = clip_imgs[i]
            base    = float(emb["clip_sim"][i])

            perturbs = _perturb_caption(
                comp.caption, comp, n=self.n, seed=42 + i)

            if perturbs:
                p_sims = self._clip_sim_batch(perturbs, img_emb)
            else:
                p_sims = [base]

            all_sims = [base] + p_sims
            std      = float(np.std(all_sims))
            rob      = float(max(0.0, 1.0 - std / 0.3))

            gen = LABEL_B_MAP.get(dp_label_b[i], "?")
            scores.append(RobustnessScore(
                sample_id        = dp_sample_ids[i],
                caption          = comp.caption,
                generator        = gen,
                split            = split,
                base_clip        = base,
                perturbed_clips  = p_sims,
                robustness_std   = std,
                robustness_score = rob,
                perturbations    = perturbs,
            ))

        elapsed = time.time() - t0
        mean_rob = float(np.mean([s.robustness_score for s in scores]))
        log.info(f"[Phase8/{split}] Done. mean_robustness={mean_rob:.4f}  "
                 f"elapsed={elapsed:.1f}s")
        return RobustnessReport(scores=scores, elapsed=elapsed, split=split)

@dataclass
class PipelineConfig:
    splits:           List[str]      = field(default_factory=lambda: ALL_SPLITS)
    max_samples:      Optional[int]  = None
    cache_dir:        str            = "./hf_cache"
    cap_pct:          float          = 85.0
    force_gpu:        bool           = True
    enable_blip:      bool           = True
    clip_weight:      float          = 0.6
    blip_weight:      float          = 0.4
    compute_kid:      bool           = True
    compute_is:       bool           = True
    train_model:      bool           = True
    resume:           bool           = False
    retrain_only:     bool           = False
    seed:             int            = 42
    output_dir:       str            = "pipeline_v5_outputs"
    batch_size:       int            = 64
    enable_semantic:  bool           = True
    enable_faithful:  bool           = True
    enable_robust:    bool           = True
    robust_n:         int            = 3

class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        global log
        log = setup_logging(cfg.output_dir, cfg.seed)
        set_global_seed(cfg.seed)
        print("\n+" + "="*70 + "+")
        print("|" + "  T2I EVALUATION PIPELINE v5.0 -- ALL SPLITS".center(70) + "|")
        print("|" + f"  cap={cfg.cap_pct}%  sem={'on' if cfg.enable_semantic else 'off'}"
              f"  faith={'on' if cfg.enable_faithful else 'off'}"
              f"  robust={'on' if cfg.enable_robust else 'off'}".center(70) + "|")
        print("+"+"="*70+"+\n")

        _has_gpu    = TORCH_OK and torch.cuda.is_available()
        _force_gpu  = cfg.force_gpu and _has_gpu
        self.rm  = ResourceManager(cap=cfg.cap_pct, force_gpu=_force_gpu,
                                   disk_path=cfg.output_dir, poll_sec=30.0)
        if PSUTIL_OK:
            du = psutil.disk_usage(cfg.output_dir
                                   if Path(cfg.output_dir).exists() else ".")
            free = du.free / 1024**3
            if free < 2.0:
                log.warning(f"[DISK] Only {free:.1f} GB free! "
                            f"Need ~2-5 GB for embeddings. "
                            f"Use --output-dir on a drive with more space.")
            elif free < 10.0:
                log.warning(f"[DISK] {free:.1f} GB free. "
                            f"Large splits may fill disk.")
        self.rm.print_report()
        self.reg = ModelRegistry(self.rm)
        
        self.total_phases = 0
        for _ in self.cfg.splits:
            self.total_phases += 3
            if self.cfg.enable_semantic: self.total_phases += 1
            if self.cfg.enable_faithful: self.total_phases += 1
            if self.cfg.enable_robust: self.total_phases += 1
        if self.cfg.train_model: self.total_phases += 1
        
        self.pbar = tqdm(total=self.total_phases, desc="Overall Pipeline Progress", position=0, leave=True, dynamic_ncols=True, colour='green')

    def _emb_path(self, split: str) -> Path:
        return self.out / f"p2_{split}.npz"

    def _process_split(self, split: str) -> None:
        log.info("\n" + "="*60)
        log.info(f"  PROCESSING SPLIT: {split.upper()}")
        log.info("="*60)

        dp = DataPipeline(self.rm, split=split,
                          max_samples=self.cfg.max_samples,
                          cache_dir=self.cfg.cache_dir)
        
        # Load Phase 1
        meta_path = self.out / f"p1_{split}_metadata.csv"
        if self.cfg.resume and meta_path.exists():
            log.info(f"[Phase1/{split}] Resume: skipped (found {meta_path.name})")
            dp.load() 
        else:
            dp.load()
            dp.print_statistics()
            dp.to_dataframe().to_csv(meta_path, index=False)

        # Load Phase 2
        emb_path = self._emb_path(split)
        emb = None
        if self.cfg.resume and emb_path.exists():
            log.info(f"[Phase2/{split}] Resume: loaded {emb_path.name}")
            raw = np.load(emb_path, allow_pickle=False)
            emb = {k: raw[k] for k in raw.files}
            self.pbar.update(1)
        else:
            ee  = EmbeddingEngine(self.rm, self.reg)
            bs  = min(self.cfg.batch_size, self.rm.clip_bs())
            emb = ee.embed_split(dp, batch_size=bs)
            is_interrupted = emb.pop("_interrupted", None) is not None
            del ee; gc.collect()
            if TORCH_OK and self.rm.gpu_n > 0:
                torch.cuda.empty_cache()
            
            np.savez_compressed(emb_path, **{k: v for k, v in emb.items() if v is not None})
            log.info(f"[Phase2/{split}] Saved -> {emb_path.name}")
                                       
            if is_interrupted:
                log.warning(f"Pipeline manually halted via KeyboardInterrupt. Progress saved to {emb_path.name}")
                sys.exit(130)
                
            log.info(f"[Phase2/{split}] clip_sim mean={emb['clip_sim'].mean():.4f}  shape={emb['clip_text'].shape}")
            self.pbar.update(1)

        # Phase 3
        p3_path = self.out / f"p3_{split}_alignment.csv"
        if self.cfg.resume and p3_path.exists():
            log.info(f"[Phase3/{split}] Resume: skipped (found {p3_path.name})")
            self.pbar.update(1)
        else:
            sa = SemanticAligner(self.rm, self.reg,
                                 enable_blip=self.cfg.enable_blip,
                                 clip_w=self.cfg.clip_weight,
                                 blip_w=self.cfg.blip_weight)
            p3 = sa.score_all(dp, emb)
            p3.print_summary()
            p3.to_dataframe().to_csv(p3_path, index=False)
            self.pbar.update(1)

        # Phase 4
        p4_path = self.out / f"p4_{split}_fidelity.csv"
        if self.cfg.resume and p4_path.exists():
            log.info(f"[Phase4/{split}] Resume: skipped (found {p4_path.name})")
            self.pbar.update(1)
        else:
            fe = FidelityEvaluator(self.rm,
                                   compute_kid=self.cfg.compute_kid,
                                   compute_is=self.cfg.compute_is)
            p4 = fe.evaluate(emb, split=split)
            p4.print_summary()
            p4.to_dataframe().to_csv(p4_path, index=False)
            self.pbar.update(1)

        # Phase 6
        p6_path = self.out / f"p6_{split}_semantic.csv"
        comps = None
        if self.cfg.enable_semantic:
            if self.cfg.resume and p6_path.exists():
                log.info(f"[Phase6/{split}] Resume: loaded {p6_path.name}")
                df = pd.read_csv(p6_path)
                comps = []
                for _, r in df.iterrows():
                    comps.append(SemanticComponents(
                        caption=str(r["caption"]),
                        objects=str(r["objects"]).split("|") if pd.notna(r.get("objects")) else ["(none)"],
                        attributes=str(r["attributes"]).split("|") if pd.notna(r.get("attributes")) and str(r.get("attributes")) else [],
                        relations=str(r["relations"]).split("|") if pd.notna(r.get("relations")) and str(r.get("relations")) else [],
                        raw_tokens=[]
                    ))
                self.pbar.update(1)
            else:
                if not SPACY_OK:
                    log.warning("[Phase6] spaCy not installed. pip install spacy")
                else:
                    sd = SemanticDecomposer()
                    comps = sd.decompose_all(dp.captions, split=split)
                    SemanticDecomposer.to_dataframe(comps).to_csv(p6_path, index=False)
                    log.info(f"[Phase6/{split}] Saved -> {p6_path.name}")
                self.pbar.update(1)

        # Phase 7
        p7_path = self.out / f"p7_{split}_faithfulness.csv"
        if self.cfg.enable_faithful and comps is not None:
            if self.cfg.resume and p7_path.exists():
                log.info(f"[Phase7/{split}] Resume: skipped (found {p7_path.name})")
                self.pbar.update(1)
            else:
                fe7 = FaithfulnessEvaluator(self.rm, self.reg)
                p7  = fe7.evaluate(
                    emb            = emb,
                    comps          = comps,
                    dp_captions    = dp.captions,
                    dp_label_b     = dp.label_b,
                    dp_sample_ids  = dp.sample_ids,
                    split          = split,
                )
                p7.print_summary()
                p7.to_dataframe().to_csv(p7_path, index=False)
                self.pbar.update(1)

        # Phase 8
        p8_path = self.out / f"p8_{split}_robustness.csv"
        if self.cfg.enable_robust and comps is not None:
            if self.cfg.resume and p8_path.exists():
                log.info(f"[Phase8/{split}] Resume: skipped (found {p8_path.name})")
                self.pbar.update(1)
            else:
                rt = RobustnessTester(self.rm, self.reg,
                                      n_perturbations=self.cfg.robust_n)
                p8 = rt.evaluate(
                    emb            = emb,
                    comps          = comps,
                    dp_captions    = dp.captions,
                    dp_label_b     = dp.label_b,
                    dp_sample_ids  = dp.sample_ids,
                    split          = split,
                )
                p8.print_summary()
                p8.to_dataframe().to_csv(p8_path, index=False)
                self.pbar.update(1)

        del dp; del emb; del comps; gc.collect()

    def _load_emb(self, split: str) -> Dict:
        ep  = self._emb_path(split)
        raw = np.load(ep, allow_pickle=False)
        return {k: raw[k] for k in raw.files}

    def run(self) -> Dict:
        t0  = time.time(); res = {}
        
        try:
            with self.rm.monitor("PIPELINE v5"):
                
                # Execute/Resume prior phases and serialize locally
                if not self.cfg.retrain_only:
                    for split in self.cfg.splits:
                        self._process_split(split)
                        if TORCH_OK and self.rm.gpu_n > 0:
                            torch.cuda.empty_cache()
                        gc.collect()
                else:
                    log.info("[retrain_only] Skipping generation, jumping straight to Phase 5 ...")
                    jump_val = len(self.cfg.splits) * (3 + int(self.cfg.enable_semantic) + int(self.cfg.enable_faithful) + int(self.cfg.enable_robust))
                    self.pbar.update(jump_val)

                # Phase 5: Train Evaluator
                if self.cfg.train_model:
                    log.info("\n" + "="*60)
                    log.info("  PHASE 5 -- EVALUATOR MODEL")
                    log.info("  Utilizing complete dataset for combined training...")
                    log.info("="*60)

                    combined_clip = []
                    combined_blip = []
                    combined_fid = []
                    combined_kid = []
                    combined_is = []
                    combined_faith = []
                    combined_obj = []
                    combined_attr = []
                    combined_rel = []
                    combined_rob = []
                    combined_label_a = []
                    blip_is_ok = False

                    # Aggregate disk checkpoints directly to bypass RAM limits
                    for split in self.cfg.splits:
                        ep = self._emb_path(split)
                        if not ep.exists():
                            log.warning(f"Missing {ep.name}, skipping Phase 5 append for {split}.")
                            continue
                            
                        emb = self._load_emb(split)
                        N = len(emb["label_a"])
                        
                        fid_arr  = np.full(N, -1.0, dtype=np.float32)
                        kid_arr  = np.full(N, -1.0, dtype=np.float32)
                        is_arr   = np.full(N, -1.0, dtype=np.float32)
                        blip_arr = np.full(N, np.nan, dtype=np.float32)
                        faith_arr= np.full(N, np.nan, dtype=np.float32)
                        obj_arr  = np.full(N, np.nan, dtype=np.float32)
                        attr_arr = np.full(N, np.nan, dtype=np.float32)
                        rel_arr  = np.full(N, np.nan, dtype=np.float32)
                        rob_arr  = np.full(N, np.nan, dtype=np.float32)

                        p4_csv = self.out / f"p4_{split}_fidelity.csv"
                        if p4_csv.exists():
                            df = pd.read_csv(p4_csv)
                            for i in range(N):
                                g = LABEL_B_MAP.get(int(emb["label_b"][i]), "?")
                                row = df[df.generator == g]
                                if len(row):
                                    fid_arr[i] = float(row["fid"].values[0])
                                    kid_arr[i] = float(row["kid_mean"].values[0])
                                    is_arr[i]  = float(row["is_mean"].values[0])

                        p3_csv = self.out / f"p3_{split}_alignment.csv"
                        if p3_csv.exists():
                            df = pd.read_csv(p3_csv)
                            if "blip_score" in df.columns:
                                blip_arr = df["blip_score"].fillna(np.nan).values.astype(np.float32)
                                blip_is_ok = True

                        p7_csv = self.out / f"p7_{split}_faithfulness.csv"
                        if p7_csv.exists():
                            df = pd.read_csv(p7_csv)
                            if "faithfulness" in df.columns:
                                faith_arr = df["faithfulness"].fillna(np.nan).values.astype(np.float32)
                                obj_arr   = df["obj_score"].fillna(np.nan).values.astype(np.float32)
                                attr_arr  = df["attr_score"].fillna(np.nan).values.astype(np.float32)
                                rel_arr   = df["rel_score"].fillna(np.nan).values.astype(np.float32)

                        p8_csv = self.out / f"p8_{split}_robustness.csv"
                        if p8_csv.exists():
                            df = pd.read_csv(p8_csv)
                            if "robustness_score" in df.columns:
                                rob_arr = df["robustness_score"].fillna(np.nan).values.astype(np.float32)

                        combined_clip.append(emb["clip_sim"])
                        combined_blip.append(blip_arr)
                        combined_fid.append(fid_arr)
                        combined_kid.append(kid_arr)
                        combined_is.append(is_arr)
                        combined_faith.append(faith_arr)
                        combined_obj.append(obj_arr)
                        combined_attr.append(attr_arr)
                        combined_rel.append(rel_arr)
                        combined_rob.append(rob_arr)
                        combined_label_a.append(emb["label_a"])
                        
                        del emb; gc.collect()

                    if len(combined_label_a) > 0:
                        full_clip = np.concatenate(combined_clip)
                        full_blip = np.concatenate(combined_blip)
                        full_fid = np.concatenate(combined_fid)
                        full_kid = np.concatenate(combined_kid)
                        full_is = np.concatenate(combined_is)
                        full_faith = np.concatenate(combined_faith)
                        full_obj = np.concatenate(combined_obj)
                        full_attr = np.concatenate(combined_attr)
                        full_rel = np.concatenate(combined_rel)
                        full_rob = np.concatenate(combined_rob)
                        full_label_a = np.concatenate(combined_label_a).astype(np.float32)

                        log.info(f"[Phase5] Combined full dataset: {len(full_label_a):,} samples")

                        ev = EvaluatorModel(self.rm.device)
                        ev.train(
                            clip_sim    = full_clip,
                            blip_scores = full_blip,
                            fid_arr     = full_fid,
                            kid_arr     = full_kid,
                            is_arr      = full_is,
                            rob_arr     = full_rob,
                            label_a     = full_label_a,
                        )

                        if ev.train_history:
                            plot_training_curve(ev.train_history,
                                str(self.out / "viz_training_curve.png"))

                        mm = ModelManager(str(self.out / "models"))
                        mm.save(ev, meta={
                            "splits":        self.cfg.splits,
                            "n_train":       len(full_label_a),
                            "n_val":         0,
                            "n_test":        0,
                            "dataset":       DATASET,
                            "clip_model":    self.reg._clip_name,
                            "seed":          self.cfg.seed,
                            "blip_ok":       blip_is_ok,
                            "test_accuracy": res.get("test_accuracy", -1.0),
                            "test_auc":      res.get("test_auc", -1.0),
                            "split_usage":   "full_dataset_combined=fit_and_earlystop",
                        })
                        gc.collect()
                        res["evaluator"]     = ev
                        res["model_manager"] = mm
                        log.info("[OK] Evaluator model trained on full dataset and saved.\n")
                    self.pbar.update(1)

            self.pbar.close()
            elapsed = time.time() - t0
            print("\n+" + "="*68 + "+")
            print("|" + "  PIPELINE v5.0 COMPLETE".center(68) + "|")
            print(f"|  Time   : {elapsed:.1f}s  ({elapsed/60:.1f}min)".ljust(69) + "|")
            print(f"|  Splits : {self.cfg.splits}".ljust(69) + "|")
            acc_s = f"{res['test_accuracy']:.4f}" if "test_accuracy" in res else "n/a"
            auc_s = f"{res['test_auc']:.4f}"      if "test_auc"      in res else "n/a"
            print(f"|  Test accuracy={acc_s}  AUC={auc_s}".ljust(69) + "|")
            print(f"|  OutDir : {str(self.out)}".ljust(69) + "|")
            print("+"+"="*68+"+")
            for f in sorted(self.out.rglob("*")):
                if f.is_file():
                    sz  = f.stat().st_size
                    szs = (f"{sz/1024:.0f}KB" if sz < 1024**2
                           else f"{sz/1024**2:.1f}MB")
                    print(f"  {str(f.relative_to(self.out)):<60}{szs:>8}")
            print("+"+"="*68+"+\n")
            return res
            
        except KeyboardInterrupt:
            self.pbar.close()
            log.warning("\n[PIPELINE] Aborted via user KeyboardInterrupt. Finalizing teardown.")
            sys.exit(130)

RUN_MODES: Dict[str, Dict] = {
    "dev": {
        "label":            "Dev / Smoke-test",
        "description":      "~500 samples | no BLIP | no KID/IS | no Robust | ~5-10 min",
        "max_samples":      500,
        "enable_blip":      False,
        "compute_kid":      False,
        "compute_is":       False,
        "enable_semantic":  True,
        "enable_faithful":  True,
        "enable_robust":    False,
        "batch_size":       32,
        "splits":           ["train", "validation", "test"],
    },
    "quick": {
        "label":            "Quick",
        "description":      "~2000 samples | no BLIP | KID+IS | Semantic+Faith | ~25-45 min",
        "max_samples":      2000,
        "enable_blip":      False,
        "compute_kid":      True,
        "compute_is":       True,
        "enable_semantic":  True,
        "enable_faithful":  True,
        "enable_robust":    True,
        "batch_size":       32,
        "splits":           ["train", "validation", "test"],
    },
    "limited": {
        "label":            "Limited",
        "description":      "~5000 samples | BLIP | full metrics | ~75-100 min",
        "max_samples":      5000,
        "enable_blip":      True,
        "compute_kid":      True,
        "compute_is":       True,
        "enable_semantic":  True,
        "enable_faithful":  True,
        "enable_robust":    True,
        "batch_size":       64,
        "splits":           ["train", "validation", "test"],
    },
    "full": {
        "label":            "Full (Production)",
        "description":      "All samples (42k/9k/45k) | BLIP | full metrics | 90+ min",
        "max_samples":      None,
        "enable_blip":      True,
        "compute_kid":      True,
        "compute_is":       True,
        "enable_semantic":  True,
        "enable_faithful":  True,
        "enable_robust":    True,
        "batch_size":       64,
        "splits":           ["train", "validation", "test"],
    },
}

_MODE_KEYS = list(RUN_MODES.keys())

def _print_mode_table():
    W = 76
    print("\n+" + "="*(W-2) + "+")
    print("|" + "  PIPELINE v5.0 -- RUN MODES".center(W-2) + "|")
    print("|" + "  Phases: CLIP|BLIP|FID/KID/IS|Semantic|Faithfulness|Robust".center(W-2) + "|")
    print("+"+"="*(W-2)+"+")
    for key, m in RUN_MODES.items():
        idx   = _MODE_KEYS.index(key) + 1
        label = m["label"]
        desc  = m["description"]
        line  = f"  [{idx}] {label:<22}  {desc}"
        print(f"|{line:<{W-2}}|")
    custom = "  [5] Custom          Use CLI flags (--max-samples / --no-spacy etc.)"
    print(f"|{custom:<{W-2}}|")
    print("+"+"="*(W-2)+"+")
    sizes = {k: (f"{v:,}" if v else "all") for k, v in SPLIT_SIZES.items()}
    tr  = sizes["train"]; va = sizes["validation"]; te = sizes["test"]
    szl = f"|  Full split sizes: train={tr}  val={va}  test={te}"
    print(f"{szl:<{W-1}}|")
    print("+"+"="*(W-2)+"+\n")

def _interactive_mode_select() -> Optional[str]:
    import sys
    if not sys.stdin.isatty():
        return None
    _print_mode_table()
    while True:
        try:
            ans = input("  Select run mode [1-5, default=4 full]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return None
        if ans == "" or ans == "4": return "full"
        if ans in ("1","2","3","4"): return _MODE_KEYS[int(ans)-1]
        if ans == "5": return None
        print("  Please enter 1, 2, 3, 4, or 5.")

def _apply_mode(cfg: PipelineConfig, mode_key: str) -> PipelineConfig:
    m = RUN_MODES[mode_key]
    cfg.max_samples      = m["max_samples"]
    cfg.enable_blip      = m["enable_blip"]
    cfg.compute_kid      = m["compute_kid"]
    cfg.compute_is       = m["compute_is"]
    cfg.enable_semantic  = m["enable_semantic"]
    cfg.enable_faithful  = m["enable_faithful"]
    cfg.enable_robust    = m["enable_robust"]
    cfg.batch_size       = m.get("batch_size", cfg.batch_size)
    cfg.splits           = list(m["splits"])
    desc = m["description"]
    log.info(f"[Mode] {mode_key.upper()} -- {desc}")
    return cfg

def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(
        "T2I Pipeline v5.0 -- Unified Multimodal Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=_MODE_KEYS, default=None,
                   help="Named run mode: dev|quick|limited|full. "
                        "Overrides --max-samples/--no-blip/--no-spacy etc.")
    p.add_argument("--list-modes",   action="store_true",
                   help="Print available run modes and exit.")
    p.add_argument("--splits",       nargs="+",
                   default=["train", "validation", "test"],
                   choices=["train", "validation", "test"],
                   help="Which splits to process")
    p.add_argument("--max-samples",  type=int,   default=None,
                   help="Max samples per split (None=all)")
    p.add_argument("--cap",          type=float, default=85.0,
                   help="Resource cap pct (RAM/VRAM)")
    p.add_argument("--batch-size",   type=int,   default=64,
                   help="Images per batch (lower=less RAM)")
    p.add_argument("--no-blip",      action="store_true")
    p.add_argument("--no-kid",       action="store_true")
    p.add_argument("--no-is",        action="store_true")
    p.add_argument("--no-train",      action="store_true")
    p.add_argument("--no-spacy",      action="store_true",
                   help="Disable Phase 6 semantic decomposition (no spaCy needed)")
    p.add_argument("--no-faithful",   action="store_true",
                   help="Disable Phase 7 faithfulness / hallucination evaluation")
    p.add_argument("--no-robust",     action="store_true",
                   help="Disable Phase 8 robustness testing (fastest run)")
    p.add_argument("--robust-n",      type=int, default=3,
                   help="Number of prompt perturbations per sample (default: 3)")
    p.add_argument("--resume",        action="store_true",
                   help="Skip completed phases automatically by scanning the disk")
    p.add_argument("--retrain-only",  action="store_true",
                   help="Skip all evaluation phases, retrain model purely from saved CSV files")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output-dir",   default="_oupipeline_v5tputs")
    p.add_argument("--cache-dir",    default="./hf_cache",
                   help="HuggingFace dataset cache dir (needs ~20GB)")
    p.add_argument("--cpu",          action="store_true")
    a = p.parse_args()

    if a.list_modes:
        _print_mode_table()
        sys.exit(0)

    _has_gpu   = TORCH_OK and torch.cuda.is_available()
    _force_gpu = not a.cpu and _has_gpu
    cfg = PipelineConfig(
        splits       = a.splits,
        max_samples  = a.max_samples,
        cap_pct      = a.cap,
        force_gpu    = _force_gpu,
        enable_blip  = not a.no_blip,
        compute_kid  = not a.no_kid,
        compute_is   = not a.no_is,
        train_model      = not a.no_train,
        resume           = a.resume,
        retrain_only     = a.retrain_only,
        seed             = a.seed,
        output_dir       = a.output_dir,
        cache_dir        = a.cache_dir,
        batch_size       = a.batch_size,
        enable_semantic  = not a.no_spacy,
        enable_faithful  = not a.no_faithful,
        enable_robust    = not a.no_robust,
        robust_n         = a.robust_n,
    )

    mode_key = a.mode
    if mode_key is None and a.max_samples is None and not a.no_blip:
        mode_key = _interactive_mode_select()
    if mode_key is not None:
        cfg = _apply_mode(cfg, mode_key)

    W   = 74
    lbl = RUN_MODES[mode_key]["label"] if mode_key else "Custom"
    samp = f"{cfg.max_samples:,}" if cfg.max_samples else "all"
    print("\n+" + "="*(W-2) + "+")
    print("|" + f"  MODE: {lbl}".center(W-2) + "|")
    print("|" + f"  Splits : {cfg.splits}".ljust(W-2) + "|")
    print("|" + f"  Samples: {samp} per split".ljust(W-2) + "|")
    sem = "on" if cfg.enable_semantic else "off"
    fai = "on" if cfg.enable_faithful else "off"
    rob = "on" if cfg.enable_robust   else "off"
    blp = "on" if cfg.enable_blip     else "off"
    print("|" + f"  BLIP={blp}  Semantic={sem}  Faithful={fai}  Robust={rob}".ljust(W-2) + "|")
    print("+"+"="*(W-2)+"+\n")
    return cfg

if __name__ == "__main__":
    Pipeline(parse_args()).run()