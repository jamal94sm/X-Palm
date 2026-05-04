"""
CompNet — Cross-Dataset Palmprint Recognition
==================================================
Dataset sampling constants (per-dataset, mirrors natural distributions):
  CASIA-MS  : 150 IDs × 29  +  40 IDs × 15  =  190 IDs
  Palm-Auth : natural distribution (no fixed cap)
  MPDv2     : 150 IDs × 33  +  40 IDs × 16  =  190 IDs
  XJTU      : 150 IDs × 29  +  40 IDs × 15  =  190 IDs
              images drawn near-uniformly across 4 variations:
              Flash-iPhone | Nature-iPhone | Flash-Huawei | Nature-Huawei

Combined evaluation set  (combined_evaluation_set = True)
---------------------------------------------------------
  1. Parse all four datasets → 190 selected IDs each.
  2. Hold out 20% (~38) from each dataset's 190 IDs for evaluation.
  3. Merge held-out IDs with new global sequential labels.
  4. Split each eval ID's images by combined_gallery_ratio (0.50).
  5. Remaining ~152 IDs of the TRAINING dataset → training.

  TWO CACHES guarantee identical conditions across all experiments:
    combined_eval_cache_path           (JSON)
    init_weights_{model}_nc{N}.pth    (alongside the JSON)

EER reporting
-------------
  EER_all : computed using ALL impostor pairs (unbalanced, reference)
  EER_bal : mean over 10 trials of 1:1 balanced genuine/impostor pairs (fair)
  Model selection and best-EER tracking use EER_bal.
"""

# ==============================================================
#  CONFIG  — edit only this block
# ==============================================================
CONFIG = {
    # ── Dataset selection ──────────────────────────────────────
    # Choices: "CASIA-MS" | "Palm-Auth" | "MPDv2" | "XJTU"
    "train_data"           : "Palm-Auth",
    "test_data"            : "Palm-Auth",   # used only when combined_evaluation_set=False

    # ── Dataset paths ──────────────────────────────────────────
    "casiams_data_root"    : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "palm_auth_data_root"  : "/home/pai-ng/Jamal/smartphone_data",
    "mpd_data_root"        : "/home/pai-ng/Jamal/MPDv2_mediapipe_manual_roi",
    "xjtu_data_root"       : "/home/pai-ng/Jamal/XJTU-UP",

    # ── Splitting ──────────────────────────────────────────────
    "train_subject_ratio"  : 0.80,
    "test_gallery_ratio"   : 0.10,

    # ── Palm-Auth toggle ───────────────────────────────────────
    "use_scanner"          : True,

    # ── Combined evaluation set ────────────────────────────────
    "combined_evaluation_set"  : False,
    "combined_gallery_ratio"   : 0.50,
    "combined_eval_cache_path" : "./combined_eval_cache.json",

    # ── Model ──────────────────────────────────────────────────
    "img_side"             : 128,
    "embedding_dim"        : 512,
    "dropout"              : 0.25,
    "arcface_s"            : 30.0,
    "arcface_m"            : 0.50,

    # ── Training ───────────────────────────────────────────────
    "batch_size"           : 128,
    "num_epochs"           : 100,
    "lr"                   : 0.001,
    "lr_step"              : 30,
    "lr_gamma"             : 0.8,
    "augment_factor"       : 2,

    # ── Misc ───────────────────────────────────────────────────
    "results_dir"          : "./rst_compnet",
    "random_seed"          : 42,
    "save_every"           : 50,
    "eval_every"           : 50,
    "num_workers"          : 4,
    "resume"               : False,
    "eval_only"            : False,
}
# ==============================================================

import os
import json
import math
import time
import random
import warnings
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

SEED = CONFIG["random_seed"]
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ALLOWED_SPECTRA = {"green", "ir", "yellow", "pink", "white", "blue"}

N_HIGH = 150
N_LOW  = 40

TARGET_HIGH_CASIA = 32
TARGET_LOW_CASIA  = 15
TARGET_HIGH_MPD   = 36
TARGET_LOW_MPD    = 16
TARGET_HIGH_XJTU  = 32
TARGET_LOW_XJTU   = 15

XJTU_VARIATIONS = [
    ("iPhone", "Flash"),
    ("iPhone", "Nature"),
    ("huawei", "Flash"),
    ("huawei", "Nature"),
]


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = max(init_ratio, 1e-6)
        self.kernel      = 0
        _S = 9.2 * self.init_ratio
        _F = 0.057 / self.init_ratio
        _G = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([_G]))
        self.sigma = nn.Parameter(torch.FloatTensor([_S]))
        self.theta = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_F]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _gen(self, ksize, c_in, c_out, sigma, gamma, theta, f, psi):
        half = ksize // 2; ksz = 2 * half + 1
        y0 = torch.arange(-half, half + 1).float()
        x0 = torch.arange(-half, half + 1).float()
        y  = y0.view(1,-1).repeat(c_out, c_in, ksz, 1)
        x  = x0.view(-1,1).repeat(c_out, c_in, 1, ksz)
        x  = x.to(sigma.device); y = y.to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5*((gamma*xt)**2 + yt**2) / (8*sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt + psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self._gen(self.kernel_size, self.channel_in,
                                self.channel_out, self.sigma, self.gamma,
                                self.theta, self.f, self.psi)
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor   = GaborConv2d(channel_in, n_competitor, ksize,
                                   stride, padding, init_ratio)
        self.a       = nn.Parameter(torch.FloatTensor([1]))
        self.b       = nn.Parameter(torch.FloatTensor([0]))
        self.argmax  = nn.Softmax(dim=1)
        self.conv1   = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor(x)
        x = self.argmax((x - self.b) * self.a)
        return self.conv2(self.maxpool(self.conv1(x)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine)
                    if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1 - one_hot) * cosine))
        return self.s * cosine


class CompNet(nn.Module):
    """CompNet = CB1 ∥ CB2 ∥ CB3 + FC(9708→emb_dim) + Dropout + ArcFace"""
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25):
        super().__init__()
        self.cb1  = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2  = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3  = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.fc   = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)

    def _backbone(self, x):
        x1 = self.cb1(x).flatten(1)
        x2 = self.cb2(x).flatten(1)
        x3 = self.cb3(x).flatten(1)
        return self.fc(torch.cat([x1, x2, x3], dim=1))

    def forward(self, x, y=None):
        return self.arc(self.drop(self._backbone(x)), y)

    @torch.no_grad()
    def get_embedding(self, x):
        return F.normalize(self._backbone(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()
        tensor  = tensor.view(c, h * w)
        idx     = tensor > 0; t = tensor[idx]
        tensor[idx] = t.sub_(t.mean()).div_(t.std() + 1e-6)
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASET PARSERS
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root, seed=42):
    rng     = random.Random(seed)
    id_spec = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        stem  = os.path.splitext(fname)[0]; parts = stem.split("_")
        if len(parts) < 4: continue
        id_spec[parts[0]+"_"+parts[1]][parts[2]].append(
            os.path.join(data_root, fname))

    all_ids = sorted(id_spec.keys())
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"Need {N_HIGH+N_LOW} IDs but only {len(all_ids)} available.")

    selected = sorted(rng.sample(all_ids, N_HIGH + N_LOW))
    rng.shuffle(selected)
    high_ids = selected[:N_HIGH]; low_ids = selected[N_HIGH:]

    def _sample(ident, target):
        spec_list = list(sorted(id_spec[ident].keys())); rng.shuffle(spec_list)
        n_spec = len(spec_list); base_s = target // n_spec; rem_s = target % n_spec
        chosen = []
        for j, sp in enumerate(spec_list):
            k = min(base_s + (1 if j < rem_s else 0), len(id_spec[ident][sp]))
            chosen.extend(rng.sample(id_spec[ident][sp], k))
        return chosen

    id2paths = {}
    for ident in high_ids: id2paths[ident] = _sample(ident, TARGET_HIGH_CASIA)
    for ident in low_ids:  id2paths[ident] = _sample(ident, TARGET_LOW_CASIA)

    actual = sum(len(v) for v in id2paths.values())
    hc = [len(id2paths[i]) for i in high_ids]
    lc = [len(id2paths[i]) for i in low_ids]
    print(f"  [CASIA-MS] ids={len(id2paths)}  total={actual}")
    print(f"    High ({N_HIGH}×~{TARGET_HIGH_CASIA}): "
          f"min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f}")
    print(f"    Low  ({N_LOW}×~{TARGET_LOW_CASIA}):  "
          f"min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f}")
    return id2paths


def parse_palm_auth_data(data_root, use_scanner=False):
    IMG_EXTS = {".jpg",".jpeg",".bmp",".png"}
    id2paths  = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir = os.path.join(subject_dir, "roi_perspective")
        if os.path.isdir(roi_dir):
            for fname in sorted(os.listdir(roi_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 3: continue
                id2paths[parts[0]+"_"+parts[1]].append(os.path.join(roi_dir, fname))
        if use_scanner:
            scan_dir = os.path.join(subject_dir, "roi_scanner")
            if os.path.isdir(scan_dir):
                for fname in sorted(os.listdir(scan_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                    parts = os.path.splitext(fname)[0].split("_")
                    if len(parts) < 4: continue
                    if parts[2].lower() not in ALLOWED_SPECTRA: continue
                    hand = parts[1].lower()
                    identity = subject_id + "_" + hand
                    id2paths[identity].append(os.path.join(scan_dir, fname))
    result = dict(id2paths)
    counts = [len(v) for v in result.values()]
    mode   = (f"perspective + scanner ({', '.join(sorted(ALLOWED_SPECTRA))})"
              if use_scanner else "perspective only")
    print(f"  [Palm-Auth/{mode}]")
    print(f"    ids={len(result)}  total={sum(counts)}  "
          f"per-id min/max/mean={min(counts)}/{max(counts)}/{sum(counts)/len(counts):.1f}")
    return result


def parse_mpd_data(data_root, seed=42):
    rng    = random.Random(seed)
    id_dev = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        stem  = os.path.splitext(fname)[0]; parts = stem.split("_")
        if len(parts) != 5: continue
        subject, session, device, hand_side, iteration = parts
        if device not in ("h","m") or hand_side not in ("l","r"): continue
        id_dev[subject+"_"+hand_side][device].append(os.path.join(data_root, fname))

    all_ids = list(id_dev.keys()); rng.shuffle(all_ids)
    all_ids.sort(
        key=lambda i: len(id_dev[i].get("h",[])) + len(id_dev[i].get("m",[])),
        reverse=True)

    if len(all_ids) < N_HIGH:
        raise ValueError(f"Need {N_HIGH} IDs but only {len(all_ids)} found.")
    high_ids = all_ids[:N_HIGH]

    low_cands = [i for i in all_ids[N_HIGH:]
                 if (len(id_dev[i].get("h",[])) +
                     len(id_dev[i].get("m",[]))) >= TARGET_LOW_MPD]
    if len(low_cands) < N_LOW:
        raise ValueError(f"Not enough IDs with ≥{TARGET_LOW_MPD} samples: "
                         f"found {len(low_cands)}, need {N_LOW}.")
    low_ids = low_cands[:N_LOW]

    def _sample(ident, target):
        paths = id_dev[ident].get("h",[]) + id_dev[ident].get("m",[])
        return rng.sample(paths, min(target, len(paths)))

    id2paths = {}
    for ident in high_ids: id2paths[ident] = _sample(ident, TARGET_HIGH_MPD)
    for ident in low_ids:  id2paths[ident] = _sample(ident, TARGET_LOW_MPD)

    actual   = sum(len(v) for v in id2paths.values())
    hc = [len(id2paths[i]) for i in high_ids]
    lc = [len(id2paths[i]) for i in low_ids]
    cutoff_h = (len(id_dev[high_ids[-1]].get("h",[])) +
                len(id_dev[high_ids[-1]].get("m",[])))
    cutoff_l = (len(id_dev[low_ids[-1]].get("h",[])) +
                len(id_dev[low_ids[-1]].get("m",[])))
    print(f"  [MPDv2] ids={len(id2paths)}  total={actual}")
    print(f"    High ({N_HIGH}×~{TARGET_HIGH_MPD}): "
          f"min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f} cutoff={cutoff_h}")
    print(f"    Low  ({N_LOW}×~{TARGET_LOW_MPD}):  "
          f"min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f} cutoff={cutoff_l}")
    return id2paths


def parse_xjtu_data(data_root, seed=42):
    rng      = random.Random(seed)
    IMG_EXTS = {".jpg",".jpeg",".bmp",".png"}
    id_var   = defaultdict(lambda: defaultdict(list))

    for device, condition in XJTU_VARIATIONS:
        var_dir = os.path.join(data_root, device, condition)
        if not os.path.isdir(var_dir):
            print(f"  [XJTU] WARNING: variation folder not found: {var_dir}")
            continue
        for id_folder in sorted(os.listdir(var_dir)):
            id_dir = os.path.join(var_dir, id_folder)
            if not os.path.isdir(id_dir): continue
            parts = id_folder.split("_")
            if len(parts) < 2 or parts[0].upper() not in ("L","R"): continue
            for fname in sorted(os.listdir(id_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                id_var[id_folder][(device, condition)].append(
                    os.path.join(id_dir, fname))

    all_ids = sorted(id_var.keys())
    print(f"  [XJTU] Total IDs found: {len(all_ids)}")
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"Need {N_HIGH+N_LOW} IDs but only {len(all_ids)} found "
                         f"in {data_root}")

    selected = sorted(rng.sample(all_ids, N_HIGH + N_LOW))
    rng.shuffle(selected)
    high_ids = selected[:N_HIGH]; low_ids = selected[N_HIGH:]

    def _sample_variations(ident, target):
        var_keys = list(XJTU_VARIATIONS); rng.shuffle(var_keys)
        n_var    = len(var_keys)
        base_v   = target // n_var; rem_v = target % n_var
        chosen   = []
        for j, vk in enumerate(var_keys):
            k         = base_v + (1 if j < rem_v else 0)
            available = id_var[ident].get(vk, [])
            k         = min(k, len(available))
            if k > 0:
                chosen.extend(rng.sample(available, k))
        return chosen

    id2paths = {}
    for ident in high_ids: id2paths[ident] = _sample_variations(ident, TARGET_HIGH_XJTU)
    for ident in low_ids:  id2paths[ident] = _sample_variations(ident, TARGET_LOW_XJTU)

    actual = sum(len(v) for v in id2paths.values())
    hc = [len(id2paths[i]) for i in high_ids]
    lc = [len(id2paths[i]) for i in low_ids]
    print(f"  [XJTU] ids={len(id2paths)}  total={actual}")
    print(f"    High ({N_HIGH}×~{TARGET_HIGH_XJTU}): "
          f"min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f}")
    print(f"    Low  ({N_LOW}×~{TARGET_LOW_XJTU}):  "
          f"min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f}")
    print(f"    Variations: {[f'{d}/{c}' for d,c in XJTU_VARIATIONS]}")
    return id2paths


def get_parser(dataset_name, cfg):
    name = dataset_name.strip().lower().replace("-","").replace("_","")
    seed = cfg["random_seed"]
    if name == "casiams":
        return lambda: parse_casia_ms(cfg["casiams_data_root"], seed=seed)
    elif name == "palmauth":
        return lambda: parse_palm_auth_data(cfg["palm_auth_data_root"],
                                            use_scanner=cfg.get("use_scanner", False))
    elif name == "mpdv2":
        return lambda: parse_mpd_data(cfg["mpd_data_root"], seed=seed)
    elif name == "xjtu":
        return lambda: parse_xjtu_data(cfg["xjtu_data_root"], seed=seed)
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. "
                         f"Use 'CASIA-MS', 'Palm-Auth', 'MPDv2', or 'XJTU'.")


def _ds_key(name):
    return name.strip().lower().replace("-","").replace("_","")


# ══════════════════════════════════════════════════════════════
#  COMBINED EVALUATION SET  (JSON cache)
# ══════════════════════════════════════════════════════════════

def build_combined_eval_set(cfg, seed=42):
    cache_path = cfg.get("combined_eval_cache_path", "./combined_eval_cache.json")

    if os.path.exists(cache_path):
        print(f"  Loading cached combined eval set from:\n    {cache_path}")
        with open(cache_path, "r") as f:
            data = json.load(f)
        gallery_samples = [(row[0], int(row[1])) for row in data["gallery"]]
        probe_samples   = [(row[0], int(row[1])) for row in data["probe"]]
        train_remaining = {k: {ident: paths for ident, paths in v.items()}
                           for k, v in data["train_remaining"].items()}
        print(f"  [combined eval] eval IDs={data['n_eval_ids']}  "
              f"gallery={len(gallery_samples)}  probe={len(probe_samples)}\n")
        return gallery_samples, probe_samples, train_remaining

    print("  Building combined evaluation set for the first time …")
    use_scanner   = cfg.get("use_scanner", False)
    gallery_ratio = cfg.get("combined_gallery_ratio", 0.50)

    parsed = {
        "casiams":  parse_casia_ms(cfg["casiams_data_root"], seed=seed),
        "palmauth": parse_palm_auth_data(cfg["palm_auth_data_root"],
                                         use_scanner=use_scanner),
        "mpdv2":    parse_mpd_data(cfg["mpd_data_root"], seed=seed),
        "xjtu":     parse_xjtu_data(cfg["xjtu_data_root"], seed=seed),
    }

    gallery_samples = []; probe_samples = []; train_remaining = {}
    label_offset    = 0

    for ds_key, id2paths in parsed.items():
        all_ids = sorted(id2paths.keys())
        n_held  = max(1, int(len(all_ids) * 0.20))
        rng_hold  = random.Random(seed + abs(hash(ds_key))           % 100000)
        rng_split = random.Random(seed + abs(hash(ds_key + "_split")) % 100000)
        shuffled = list(all_ids); rng_hold.shuffle(shuffled)
        held_ids = set(shuffled[:n_held])
        train_remaining[ds_key] = {k: v for k, v in id2paths.items()
                                   if k not in held_ids}
        n_gal_ds = 0; n_prob_ds = 0
        for local_idx, ident in enumerate(sorted(held_ids)):
            global_label = label_offset + local_idx
            paths = list(id2paths[ident]); rng_split.shuffle(paths)
            n_gal = max(1, int(len(paths) * gallery_ratio))
            for p in paths[:n_gal]:
                gallery_samples.append((p, global_label)); n_gal_ds += 1
            for p in paths[n_gal:]:
                probe_samples.append((p, global_label));   n_prob_ds += 1
        print(f"  [{ds_key}] total={len(all_ids)} IDs  held-out={n_held}  "
              f"train={len(train_remaining[ds_key])}  "
              f"gallery={n_gal_ds}  probe={n_prob_ds}")
        label_offset += n_held

    print(f"  [combined eval] total eval IDs={label_offset}  "
          f"gallery={len(gallery_samples)}  probe={len(probe_samples)}")

    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "gallery":         [[p, l] for p, l in gallery_samples],
            "probe":           [[p, l] for p, l in probe_samples],
            "train_remaining": {k: {ident: paths for ident, paths in v.items()}
                                for k, v in train_remaining.items()},
            "n_eval_ids":      label_offset,
            "seed":            seed,
            "use_scanner":     use_scanner,
            "gallery_ratio":   gallery_ratio,
        }, f, indent=2)
    print(f"  Eval set cached to:\n    {cache_path}\n")
    return gallery_samples, probe_samples, train_remaining


# ══════════════════════════════════════════════════════════════
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(net, cfg, num_classes, device):
    cache_path   = cfg.get("combined_eval_cache_path", "./combined_eval_cache.json")
    cache_dir    = os.path.dirname(os.path.abspath(cache_path))
    model_name   = type(net.module if isinstance(net, DataParallel) else net).__name__
    weights_path = os.path.join(cache_dir,
                                f"init_weights_{model_name}_nc{num_classes}.pth")
    _net = net.module if isinstance(net, DataParallel) else net
    if os.path.exists(weights_path):
        print(f"  Loading cached init weights from:\n    {weights_path}")
        _net.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"  Saving init weights to:\n    {weights_path}")
        torch.save(_net.state_dict(), weights_path)
    return net


# ══════════════════════════════════════════════════════════════
#  SPLITS
# ══════════════════════════════════════════════════════════════

def split_same_dataset(id2paths, train_subject_ratio=0.80,
                       gallery_ratio=0.50, seed=42):
    rng        = random.Random(seed)
    identities = sorted(id2paths.keys()); rng.shuffle(identities)
    n_train    = max(1, int(len(identities) * train_subject_ratio))
    train_ids  = identities[:n_train]; test_ids = identities[n_train:]
    train_label_map = {k: i for i, k in enumerate(train_ids)}
    test_label_map  = {k: i for i, k in enumerate(test_ids)}
    train_samples   = [(p, train_label_map[ident])
                       for ident in train_ids for p in id2paths[ident]]
    gallery_samples, probe_samples = [], []
    for ident in test_ids:
        paths = list(id2paths[ident]); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p, test_label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p, test_label_map[ident]))
    return train_samples, gallery_samples, probe_samples, train_label_map, test_label_map


def split_cross_dataset_test(id2paths, gallery_ratio=0.50, seed=42):
    rng       = random.Random(seed)
    label_map = {k: i for i, k in enumerate(sorted(id2paths.keys()))}
    gallery_samples, probe_samples = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p, label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p, label_map[ident]))
    return gallery_samples, probe_samples, label_map


# ══════════════════════════════════════════════════════════════
#  PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════

class SingleDataset(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([
            T.Resize(img_side), T.ToTensor(), NormSingleROI(outchannels=1)])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


class AugmentedDataset(Dataset):
    def __init__(self, samples, img_side=128, augment_factor=1):
        self.samples        = samples
        self.augment_factor = augment_factor
        self.aug_transform  = T.Compose([
            T.Resize(img_side),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1),
                T.RandomChoice([
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.5*img_side, 0.0)),
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.0, 0.5*img_side)),
                ]),
            ]),
            T.ToTensor(), NormSingleROI(outchannels=1),
        ])
    def __len__(self): return len(self.samples) * self.augment_factor
    def __getitem__(self, index):
        real_idx    = index % len(self.samples)
        path, label = self.samples[real_idx]
        return self.aug_transform(Image.open(path).convert("L")), label


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, optimizer, device, phase):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if is_train: optimizer.zero_grad()
            output = model(data, target if is_train else None)
            loss   = criterion(output, target)
            if is_train: loss.backward(); optimizer.step()
            running_loss    += loss.item() * data.size(0)
            running_correct += output.data.max(1)[1].eq(target).sum().item()
            total           += data.size(0)
    return running_loss / max(total, 1), 100.0 * running_correct / max(total, 1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval(); feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _single_eer(genuine, impostor):
    """Compute EER from two score arrays. Flips direction if needed."""
    if genuine.mean() < impostor.mean():
        genuine = -genuine; impostor = -impostor
    y   = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    s   = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)


def compute_eer(scores_array, n_trials=10, seed=42):
    """
    Returns two EER values:
      eer_all : EER computed using ALL impostor pairs (unbalanced)
      eer_bal : mean EER over n_trials of 1:1 balanced genuine/impostor sampling
    """
    rng  = np.random.RandomState(seed)
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0:
        return 1.0, 1.0

    # ── EER with ALL impostor pairs ───────────────────────────
    eer_all = _single_eer(ins.copy(), outs.copy())

    # ── EER with balanced 1:1 sampled impostor pairs ──────────
    n_imp = min(len(ins), len(outs))   # 1:1 ratio
    eers  = []
    for _ in range(n_trials):
        imp_sample = rng.choice(outs, size=n_imp, replace=False)
        eers.append(_single_eer(ins.copy(), imp_sample))
    eer_bal = float(np.mean(eers))

    return eer_all, eer_bal


def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    """
    Returns (eer_all, eer_bal, rank1).
      eer_all : EER with all impostor pairs
      eer_bal : EER with balanced 1:1 impostor sampling (mean over 10 trials)
      rank1   : Rank-1 identification accuracy (%)
    """
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe = len(probe_feats)

    sim_matrix = probe_feats @ gallery_feats.T

    scores_list, labels_list = [], []
    for i in range(n_probe):
        for j in range(sim_matrix.shape[1]):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr       = np.column_stack([scores_list, labels_list])
    eer_all, eer_bal = compute_eer(scores_arr)

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]]
                  for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")
    _save_roc_det(scores_arr, out_dir, tag)

    print(f"  [{tag}]  "
          f"EER_all={eer_all*100:.4f}%  EER_bal={eer_bal*100:.4f}%  "
          f"Rank-1={rank1:.2f}%")
    return eer_all, eer_bal, rank1


def _save_roc_det(scores_arr, out_dir, tag):
    ins  = scores_arr[scores_arr[:, 1] ==  1, 0]
    outs = scores_arr[scores_arr[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thr = roc_curve(y, s, pos_label=1); fnr = 1 - tpr
    try:
        pdf = PdfPages(os.path.join(out_dir, f"roc_det_{tag}.pdf"))
        for (xd, yd, xl, yl, title, xlim, ylim) in [
            (fpr*100, tpr*100, "FAR (%)", "GAR (%)", f"ROC — {tag}", [0,5], [90,100]),
            (fpr*100, fnr*100, "FAR (%)", "FRR (%)", f"DET — {tag}", [0,5], [0,5]),
        ]:
            fig, ax = plt.subplots()
            ax.plot(xd, yd, 'b-^', markersize=2)
            ax.plot(np.linspace(0,100,101),
                    np.linspace(100,0,101) if "ROC" in title else np.linspace(0,100,101),
                    'k-')
            ax.set(xlim=xlim, ylim=ylim, xlabel=xl, ylabel=yl, title=title)
            ax.grid(True); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(thr, fpr*100, 'r-.', label='FAR', markersize=2)
        ax.plot(thr, fnr*100, 'b-^', label='FRR', markersize=2)
        ax.set(xlabel="Threshold", ylabel="Rate (%)", title=f"FAR/FRR — {tag}")
        ax.legend(); ax.grid(True); pdf.savefig(fig); plt.close(fig)
        pdf.close()
    except Exception as e:
        print(f"  [warn] plot failed: {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    train_data          = CONFIG["train_data"]
    test_data           = CONFIG["test_data"]
    test_gallery_ratio  = CONFIG["test_gallery_ratio"]
    train_subject_ratio = CONFIG["train_subject_ratio"]
    results_dir         = CONFIG["results_dir"]
    img_side            = CONFIG["img_side"]
    batch_size          = CONFIG["batch_size"]
    num_epochs          = CONFIG["num_epochs"]
    lr                  = CONFIG["lr"]
    lr_step             = CONFIG["lr_step"]
    lr_gamma            = CONFIG["lr_gamma"]
    dropout             = CONFIG["dropout"]
    arcface_s           = CONFIG["arcface_s"]
    arcface_m           = CONFIG["arcface_m"]
    embedding_dim       = CONFIG["embedding_dim"]
    augment_factor      = CONFIG["augment_factor"]
    seed                = CONFIG["random_seed"]
    save_every          = CONFIG["save_every"]
    eval_every          = CONFIG["eval_every"]
    nw                  = CONFIG["num_workers"]
    use_scanner         = CONFIG.get("use_scanner", False)
    use_combined_eval   = CONFIG.get("combined_evaluation_set", False)

    same_dataset = (not use_combined_eval and
                    _ds_key(train_data) == _ds_key(test_data))

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  CompNet Palmprint Recognition")
    print(f"  Device         : {device}")
    print(f"  Train dataset  : {train_data}")
    print(f"  Evaluation     : "
          f"{'combined (4 datasets, cached)' if use_combined_eval else test_data}")
    if use_combined_eval:
        print(f"  Cache path     : {CONFIG.get('combined_eval_cache_path','')}")
    if same_dataset:
        print(f"  Mode           : same-dataset split "
              f"({int(train_subject_ratio*100)}% train / "
              f"{int((1-train_subject_ratio)*100)}% test)")
    if "palm-auth" in train_data.lower() or use_combined_eval:
        print(f"  Scanner data   : "
              f"{'ON  ('+', '.join(sorted(ALLOWED_SPECTRA))+')' if use_scanner else 'OFF'}")
    print(f"  Sampling       : CASIA {N_HIGH}×{TARGET_HIGH_CASIA}+{N_LOW}×{TARGET_LOW_CASIA}  "
          f"MPD {N_HIGH}×{TARGET_HIGH_MPD}+{N_LOW}×{TARGET_LOW_MPD}  "
          f"XJTU {N_HIGH}×{TARGET_HIGH_XJTU}+{N_LOW}×{TARGET_LOW_XJTU}")
    print(f"  Augment factor : {augment_factor}×")
    print(f"{'='*60}\n")

    # ── combined evaluation set ───────────────────────────────────────────
    train_remaining = {}
    if use_combined_eval:
        gallery_samples, probe_samples, train_remaining = \
            build_combined_eval_set(CONFIG, seed=seed)
        eval_tag_base = "combined"
    else:
        eval_tag_base = test_data.replace("-","")

    # ── training data ─────────────────────────────────────────────────────
    if use_combined_eval:
        train_id2paths = train_remaining[_ds_key(train_data)]
        print(f"Scanning {train_data} (training portion, eval IDs excluded) …")
        n_train_ids  = len(train_id2paths)
        n_train_imgs = sum(len(v) for v in train_id2paths.values())
        print(f"  {n_train_ids} identities, {n_train_imgs} images.\n")
        train_label_map = {k: i for i, k in enumerate(sorted(train_id2paths))}
        train_samples   = [(p, train_label_map[ident])
                           for ident, paths in train_id2paths.items()
                           for p in paths]
        num_classes = len(train_label_map)
        n_test_ids  = len(set(l for _, l in gallery_samples + probe_samples))
        n_test_imgs = len(gallery_samples) + len(probe_samples)

    elif same_dataset:
        print(f"Scanning {train_data} (shared train+test) …")
        all_id2paths = get_parser(train_data, CONFIG)()
        n_total_ids  = len(all_id2paths)
        n_total_imgs = sum(len(v) for v in all_id2paths.values())
        print(f"  Found {n_total_ids} identities, {n_total_imgs} images.\n")
        (train_samples, gallery_samples, probe_samples,
         train_label_map, _) = split_same_dataset(
            all_id2paths, train_subject_ratio=train_subject_ratio,
            gallery_ratio=test_gallery_ratio, seed=seed)
        num_classes  = len(train_label_map)
        n_train_ids  = num_classes
        n_train_imgs = len(train_samples)
        n_test_ids   = n_total_ids - n_train_ids
        n_test_imgs  = len(gallery_samples) + len(probe_samples)

    else:
        print(f"Scanning {train_data} (train) …")
        train_id2paths = get_parser(train_data, CONFIG)()
        n_train_ids    = len(train_id2paths)
        n_train_imgs   = sum(len(v) for v in train_id2paths.values())
        print(f"  Found {n_train_ids} identities, {n_train_imgs} images.\n")
        train_label_map = {k: i for i, k in enumerate(sorted(train_id2paths))}
        train_samples   = [(p, train_label_map[ident])
                           for ident, paths in train_id2paths.items()
                           for p in paths]
        num_classes = len(train_label_map)
        print(f"Scanning {test_data} (test) …")
        test_id2paths = get_parser(test_data, CONFIG)()
        n_test_ids    = len(test_id2paths)
        n_test_imgs   = sum(len(v) for v in test_id2paths.values())
        print(f"  Found {n_test_ids} identities, {n_test_imgs} images.\n")
        gallery_samples, probe_samples, _ = split_cross_dataset_test(
            test_id2paths, gallery_ratio=test_gallery_ratio, seed=seed)

    # ── data loaders ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        AugmentedDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Train  : {n_train_ids} subjects | "
          f"{n_train_imgs} imgs (+aug → {n_train_imgs*augment_factor})")
    print(f"  Eval   : {n_test_ids} subjects | "
          f"Gallery {len(gallery_samples)} | Probe {len(probe_samples)}")
    print(f"  Classes: {num_classes}\n")

    # ── model ─────────────────────────────────────────────────────────────
    print(f"Building CompNet — num_classes={num_classes} …")
    net = CompNet(num_classes, embedding_dim=embedding_dim,
                  arcface_s=arcface_s, arcface_m=arcface_m, dropout=dropout)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        net = DataParallel(net)

    if use_combined_eval:
        net = get_or_create_init_weights(net, CONFIG, num_classes, device)
    else:
        print("  Training from scratch (random init).")

    if CONFIG.get("resume", False):
        for ckpt in ["net_params_best_eer.pth", "net_params_best.pth", "net_params.pth"]:
            path = os.path.join(results_dir, ckpt)
            if os.path.exists(path):
                _net = net.module if isinstance(net, DataParallel) else net
                _net.load_state_dict(torch.load(path, map_location=device))
                print(f"  Resumed from : {path}"); break
        else:
            print("  No checkpoint found — starting from init weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

    # ── training state variables ───────────────────────────────────────────
    train_losses, train_accs = [], []
    # best-model selection uses EER_bal (balanced, fair)
    best_eer     = 1.0
    last_eer_all = float("nan")   # EER with all impostors
    last_eer_bal = float("nan")   # EER with balanced 1:1 sampling
    last_rank1   = float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 evaluated every {eval_every} epochs.")
    print(f"  EER_all = all impostor pairs | EER_bal = balanced 1:1 sampling\n")

    if CONFIG.get("eval_only", False):
        print("  eval_only=True — skipping training.\n")
    else:
        # ── pre-training evaluation ────────────────────────────────────────
        _net = net.module if isinstance(net, DataParallel) else net
        print("  Pre-training evaluation (before any gradient update) …")
        cur_eer_all, cur_eer_bal, cur_rank1 = evaluate(
            _net, probe_loader, gallery_loader,
            device, out_dir=rst_eval, tag=f"ep-001_pretrain_{eval_tag_base}")
        best_eer     = cur_eer_bal   # use balanced EER for model selection
        last_eer_all = cur_eer_all
        last_eer_bal = cur_eer_bal
        last_rank1   = cur_rank1
        torch.save(_net.state_dict(),
                   os.path.join(results_dir, "net_params_best_eer.pth"))
        print(f"  *** Initial best EER_bal: {best_eer*100:.4f}% ***\n")

        for epoch in range(num_epochs):
            t_loss, t_acc = run_one_epoch(
                net, train_loader, criterion, optimizer, device, "training")
            scheduler.step()

            train_losses.append(t_loss); train_accs.append(t_acc)
            _net = net.module if isinstance(net, DataParallel) else net

            if (epoch % eval_every == 0 and epoch > 0) or epoch == num_epochs - 1:
                tag = f"ep{epoch:04d}_{eval_tag_base}"
                cur_eer_all, cur_eer_bal, cur_rank1 = evaluate(
                    _net, probe_loader, gallery_loader,
                    device, out_dir=rst_eval, tag=tag)
                last_eer_all = cur_eer_all
                last_eer_bal = cur_eer_bal
                last_rank1   = cur_rank1
                if cur_eer_bal < best_eer:
                    best_eer = cur_eer_bal
                    torch.save(_net.state_dict(),
                               os.path.join(results_dir, "net_params_best_eer.pth"))
                    print(f"  *** New best EER_bal: {best_eer*100:.4f}% ***")

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                ts = time.strftime("%H:%M:%S")
                if math.isnan(last_eer_all):
                    eer_str = "N/A"
                else:
                    eer_str = (f"EER_all={last_eer_all*100:.4f}% | "
                               f"EER_bal={last_eer_bal*100:.4f}%")
                rank1_str = f"{last_rank1:.2f}%" if not math.isnan(last_rank1) else "N/A"
                print(f"[{ts}] ep {epoch:04d} | loss={t_loss:.5f} | acc={t_acc:.2f}% | "
                      f"{eer_str} | Rank-1={rank1_str}")

            if epoch % save_every == 0 or epoch == num_epochs - 1:
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params.pth"))
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].plot(train_losses,'b'); axes[0].set_title("Train Loss")
                    axes[0].set_xlabel("epoch"); axes[0].grid(True)
                    axes[1].plot(train_accs,  'b'); axes[1].set_title("Train Acc (%)")
                    axes[1].set_xlabel("epoch"); axes[1].grid(True)
                    fig.tight_layout()
                    fig.savefig(os.path.join(results_dir, "train_curves.png"))
                    plt.close(fig)
                except Exception:
                    pass

    # ── final evaluation ──────────────────────────────────────────────────
    print(f"\n=== Final evaluation "
          f"({'combined' if use_combined_eval else test_data}) ===")
    best_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(results_dir, "net_params.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))

    saved_name = (f"CompNet_train{train_data.replace('-','').replace(' ','')}"
                  f"_eval{eval_tag_base}.pth")
    torch.save(eval_net.state_dict(), os.path.join(results_dir, saved_name))
    print(f"  Model saved as {saved_name}")

    final_eer_all, final_eer_bal, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag=f"FINAL_{eval_tag_base}")

    print(f"\n{'='*60}")
    print(f"  Train  : {train_data} ({n_train_ids} subjects, {n_train_imgs} imgs)")
    print(f"  Eval   : "
          f"{'combined (CASIA-MS + Palm-Auth + MPDv2 + XJTU)' if use_combined_eval else test_data}")
    print(f"  FINAL EER_all  : {final_eer_all*100:.4f}%")
    print(f"  FINAL EER_bal  : {final_eer_bal*100:.4f}%")
    print(f"  FINAL Rank-1   : {final_rank1:.3f}%")
    print(f"  Results        : {results_dir}")
    print(f"{'='*60}\n")

    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Train dataset      : {train_data}\n")
        f.write(f"Train subjects     : {n_train_ids}\n")
        f.write(f"Train images       : {n_train_imgs}\n")
        f.write(f"Augment factor     : {augment_factor}×\n")
        f.write(f"Scanner data       : {use_scanner}\n")
        if use_scanner:
            f.write(f"Scanner spectra    : {', '.join(sorted(ALLOWED_SPECTRA))}\n")
        f.write(f"Combined eval      : {use_combined_eval}\n")
        if use_combined_eval:
            f.write(f"Eval cache         : "
                    f"{CONFIG.get('combined_eval_cache_path','')}\n")
        f.write(f"Sampling CASIA-MS  : {N_HIGH}×{TARGET_HIGH_CASIA} + "
                f"{N_LOW}×{TARGET_LOW_CASIA}\n")
        f.write(f"Sampling MPDv2     : {N_HIGH}×{TARGET_HIGH_MPD} + "
                f"{N_LOW}×{TARGET_LOW_MPD}\n")
        f.write(f"Sampling XJTU      : {N_HIGH}×{TARGET_HIGH_XJTU} + "
                f"{N_LOW}×{TARGET_LOW_XJTU}\n")
        f.write(f"Num classes        : {num_classes}\n")
        f.write(f"Eval set           : "
                f"{'combined' if use_combined_eval else test_data}\n")
        f.write(f"Eval subjects      : {n_test_ids}\n")
        f.write(f"Gallery samples    : {len(gallery_samples)}\n")
        f.write(f"Probe samples      : {len(probe_samples)}\n")
        f.write(f"Final EER_all      : {final_eer_all*100:.6f}%\n")
        f.write(f"Final EER_bal      : {final_eer_bal*100:.6f}%\n")
        f.write(f"Final Rank-1       : {final_rank1:.3f}%\n")


if __name__ == "__main__":
    main()
