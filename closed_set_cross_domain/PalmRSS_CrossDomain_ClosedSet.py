"""
PalmRSS — Single Source Domain Generalization for Palm Biometrics
Paper: Jia et al., Pattern Recognition 2025
https://doi.org/10.1016/j.patcog.2025.111620

Adapted for Palm-Auth dataset with cross-domain closed-set evaluation.

Model, training method, hyperparameters: unchanged from original PalmRSS.
Dataset      : Palm-Auth (roi_perspective + roi_scanner)
Images       : loaded as grayscale (convert("L")) — consistent with CCNet design
Evaluation   : Cross-Domain Closed-Set (12 settings)
               gallery vs probe → EER + Rank-1
Checkpoint   : saved by best Rank-1

D1 / D2 split:
  The training domain is split 50/50 per identity:
    D1 = first half of each identity's images (simulates session 1)
    D2 = second half of each identity's images (simulates session 2)
  The adversarial loss L_adv aligns D1 and D2 feature distributions.
  FAT + HM alignment is applied between D1 and D2 pairs.

Settings (12 total)
────────────────────
  S_scanner         │ Train : roi_perspective (all conditions, 190 IDs)
                    │ Gallery: 50% of scanner samples  (148 shared IDs)
                    │ Probe  : 50% of scanner samples  (148 shared IDs)

  S_scanner_to_persp│ Train : roi_scanner (148 IDs)
                    │ Gallery: 50% of perspective samples (148 shared IDs)
                    │ Probe  : 50% of perspective samples (148 shared IDs)

  S_(A,B) (×10)     │ Train : perspective (all except A and B) + scanner
                    │ Gallery: ALL condition A images
                    │ Probe  : ALL condition B images

Gallery/probe splits saved to palm_auth_closedset_splits.json on first run.
"""

import copy
import json
import math
import os
import random
import time
import warnings
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn import metrics
from sklearn.metrics import auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS  (unchanged from original PalmRSS)
# ============================================================

PALM_AUTH_ROOT  = "/home/pai-ng/Jamal/smartphone_data"
SCANNER_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

# Evaluation
TEST_GALLERY_RATIO = 0.50
SPLITS_FILE        = "./palm_auth_closedset_splits.json"
SAVE_DIR           = "./rst_palmrss_crossdomain"

# Paired conditions
PAIRED_CONDITIONS = [
    ("wet",  "text"),
    ("wet",  "rnd"),
    ("rnd",  "text"),
    ("sf",   "roll"),
    ("jf",   "pitch"),
    ("bf",   "far"),
    ("roll", "close"),
    ("far",  "jf"),
    ("fl",   "sf"),
    ("roll", "pitch"),
]

# Architecture
COM_WEIGHT  = 0.8
ARC_S       = 30.0
ARC_M       = 0.5
FC_DIM1     = 4096
FC_DIM2     = 2048
DROPOUT     = 0.5

# Loss weights
W_CE        = 0.8
W_CON       = 0.1
W_SIM       = 0.1
LAMBDA_HYB  = 1.0
TEMPERATURE = 0.07
BASE_TEMP   = 0.07

# FDA
BETA        = 0.1

# Training
BATCH_SIZE  = 32
EPOCH_NUM   = 250
LR          = 0.001
LR_STEP     = 500
LR_GAMMA    = 0.8
IMSIDE      = 128

# Logging
PRINT_INTERVAL = 10
SAVE_INTERVAL  = 50

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# DATA COLLECTION HELPERS
# ============================================================

def _collect_perspective(data_root):
    """condition → identity → [path, ...]"""
    cond_paths = defaultdict(lambda: defaultdict(list))
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir = os.path.join(subject_dir, "roi_perspective")
        if not os.path.isdir(roi_dir): continue
        for fname in sorted(os.listdir(roi_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 3: continue
            identity  = parts[0] + "_" + parts[1].lower()
            condition = parts[2].lower()
            cond_paths[condition][identity].append(os.path.join(roi_dir, fname))
    return cond_paths


def _collect_scanner(data_root, scanner_spectra):
    """identity → [path, ...]"""
    scanner_paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        scan_dir = os.path.join(subject_dir, "roi_scanner")
        if not os.path.isdir(scan_dir): continue
        for fname in sorted(os.listdir(scan_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 4: continue
            if parts[2].lower() not in scanner_spectra: continue
            identity = parts[0] + "_" + parts[1].lower()
            scanner_paths[identity].append(os.path.join(scan_dir, fname))
    return scanner_paths


def _all_samples(id2paths, label_map):
    return [(p, label_map[ident])
            for ident, paths in id2paths.items()
            for p in paths]


def _split_d1_d2(id2paths, label_map, seed):
    """
    Split training samples 50/50 per identity into D1 and D2.
    D1 = first half (simulates session 1 — earlier collection time)
    D2 = second half (simulates session 2 — later  collection time)
    """
    rng = random.Random(seed)
    d1, d2 = [], []
    for ident in sorted(id2paths.keys()):
        paths = list(id2paths[ident])
        rng.shuffle(paths)
        half = max(1, len(paths) // 2)
        for p in paths[:half]: d1.append((p, label_map[ident]))
        for p in paths[half:]: d2.append((p, label_map[ident]))
    return d1, d2


# ============================================================
# GALLERY/PROBE SPLIT PERSISTENCE
# ============================================================

def _make_gallery_probe_split(id2paths, gallery_ratio, rng):
    stored = {}
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        n_gal = min(n_gal, len(paths) - 1) if len(paths) > 1 else len(paths)
        stored[ident] = {
            "gallery": paths[:n_gal],
            "probe"  : paths[n_gal:] if len(paths) > 1 else paths
        }
    return stored


def _gallery_probe_split_from_stored(id2paths, label_map, stored_split):
    gallery, probe = [], []
    for ident, path_sets in stored_split.items():
        if ident not in label_map: continue
        label = label_map[ident]
        for p in path_sets["gallery"]: gallery.append((p, label))
        for p in path_sets["probe"]:   probe.append((p, label))
    return gallery, probe


def generate_closedset_splits(cond_paths, scanner_paths, gallery_ratio, seed):
    rng = random.Random(seed)
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    scanner_ids = sorted(scanner_paths.keys())
    splits = {}
    splits["S_scanner"] = _make_gallery_probe_split(
        {i: scanner_paths[i] for i in scanner_ids}, gallery_ratio, rng)
    splits["S_scanner_to_persp"] = _make_gallery_probe_split(
        {i: persp_all[i] for i in scanner_ids}, gallery_ratio, rng)
    return splits


def load_or_generate_closedset_splits(cond_paths, scanner_paths,
                                      gallery_ratio, seed):
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
        log(f"Loaded existing gallery/probe splits from: {SPLITS_FILE}")
    else:
        log(f"Generating gallery/probe splits (seed={seed}) → {SPLITS_FILE}")
        splits = generate_closedset_splits(
            cond_paths, scanner_paths, gallery_ratio, seed)
        with open(SPLITS_FILE, "w") as f:
            json.dump(splits, f, indent=2)
        log(f"Splits saved to: {SPLITS_FILE}")
    for key, val in splits.items():
        n_gal = sum(len(v["gallery"]) for v in val.values())
        n_prb = sum(len(v["probe"])   for v in val.values())
        log(f"  {key:<30}  IDs={len(val)}  gallery={n_gal}  probe={n_prb}")
    return splits


# ============================================================
# PARSERS — return (d1, d2, gallery, probe, num_classes)
# ============================================================

def parse_setting_scanner(cond_paths, scanner_paths, stored_splits, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    all_persp_ids = sorted(persp_all.keys())
    scanner_ids   = sorted(scanner_paths.keys())
    train_label_map = {ident: i for i, ident in enumerate(all_persp_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    num_train_cls   = len(all_persp_ids)
    d1, d2 = _split_d1_d2(
        {i: persp_all[i] for i in all_persp_ids}, train_label_map, seed)
    split = stored_splits["S_scanner"]
    gallery, probe = _gallery_probe_split_from_stored(
        {i: scanner_paths[i] for i in scanner_ids}, test_label_map, split)
    _print_stats("S_scanner", len(all_persp_ids), len(scanner_ids),
                 len(d1), len(d2), len(gallery), len(probe))
    return d1, d2, gallery, probe, num_train_cls


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths,
                                         stored_splits, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    scanner_ids = sorted(scanner_paths.keys())
    train_label_map = {ident: i for i, ident in enumerate(scanner_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    num_train_cls   = len(scanner_ids)
    d1, d2 = _split_d1_d2(scanner_paths, train_label_map, seed)
    split = stored_splits["S_scanner_to_persp"]
    gallery, probe = _gallery_probe_split_from_stored(
        {i: persp_all[i] for i in scanner_ids}, test_label_map, split)
    _print_stats("S_scanner_to_persp", len(scanner_ids), len(scanner_ids),
                 len(d1), len(d2), len(gallery), len(probe))
    return d1, d2, gallery, probe, num_train_cls


def parse_setting_paired_conditions(cond_a, cond_b, cond_paths,
                                    scanner_paths, seed):
    paths_a = cond_paths.get(cond_a, {})
    paths_b = cond_paths.get(cond_b, {})
    if not paths_a: raise ValueError(f"No images for condition '{cond_a}'")
    if not paths_b: raise ValueError(f"No images for condition '{cond_b}'")
    eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
    if not eligible_ids:
        raise ValueError(f"No IDs with both '{cond_a}' and '{cond_b}'")
    label_map   = {ident: i for i, ident in enumerate(eligible_ids)}
    num_classes = len(eligible_ids)

    # Build full training id2paths (all perspective except A/B + scanner)
    train_id2paths = defaultdict(list)
    for cond, cond_dict in cond_paths.items():
        if cond in (cond_a, cond_b): continue
        for ident in eligible_ids:
            train_id2paths[ident].extend(cond_dict.get(ident, []))
    for ident in eligible_ids:
        train_id2paths[ident].extend(scanner_paths.get(ident, []))

    d1, d2 = _split_d1_d2(train_id2paths, label_map, seed)
    gallery = _all_samples({i: paths_a[i] for i in eligible_ids}, label_map)
    probe   = _all_samples({i: paths_b[i] for i in eligible_ids}, label_map)

    _print_stats(f"S_{cond_a}_{cond_b}", num_classes, num_classes,
                 len(d1), len(d2), len(gallery), len(probe))
    return d1, d2, gallery, probe, num_classes


def _print_stats(name, n_train, n_test, d1_n, d2_n, gallery_n, probe_n):
    log(f"  [{name}]")
    log(f"    Train IDs / Test IDs : {n_train} / {n_test}")
    log(f"    D1 / D2              : {d1_n} / {d2_n}")
    log(f"    Gallery / Probe      : {gallery_n} / {probe_n}")


# ============================================================
# DATASET  (unchanged from original PalmRSS)
# Images loaded as grayscale — consistent with CCNet/Gabor design
# ============================================================

class NormSingleROI:
    def __init__(self, outchannels=1): self.outchannels = outchannels
    def __call__(self, tensor):
        c, h, w = tensor.size()
        flat = tensor.view(c, h * w)
        idx  = flat > 0; t = flat[idx]
        if t.numel() > 1:
            flat[idx] = (t - t.mean()) / (t.std() + 1e-6)
        tensor = flat.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


class PalmDataset(Dataset):
    def __init__(self, samples, train=True, imside=IMSIDE):
        self.samples = samples
        self.train   = train
        self.labels  = [s[1] for s in samples]

        if train:
            self.tf = T.Compose([
                T.Resize(imside),
                T.RandomChoice([
                    T.ColorJitter(brightness=0, contrast=0.05),
                    T.RandomResizedCrop(imside, scale=(0.8, 1.0),
                                        ratio=(1., 1.)),
                    T.RandomPerspective(distortion_scale=0.15, p=1),
                    T.RandomChoice([
                        T.RandomRotation(10, expand=False,
                                         center=(int(0.5*imside), 0)),
                        T.RandomRotation(10, expand=False,
                                         center=(0, int(0.5*imside))),
                    ]),
                ]),
                T.ToTensor(), NormSingleROI(1),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(imside), T.ToTensor(), NormSingleROI(1)])

    def __len__(self): return len(self.samples)

    def _load(self, idx):
        path, label = self.samples[idx]
        return self.tf(Image.open(path).convert("L")), label

    def __getitem__(self, idx):
        img1, label = self._load(idx)
        same = [i for i, l in enumerate(self.labels) if l == label]
        idx2 = idx
        if self.train and len(same) > 1:
            while idx2 == idx:
                idx2 = int(np.random.choice(same))
        img2, _ = self._load(idx2)
        return (img1, img2), label


# ============================================================
# MODEL — CCNet 2-channel input  (unchanged from original PalmRSS)
# ============================================================

class GaborConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize, stride=1,
                 padding=0, init_ratio=1.):
        super().__init__()
        r = init_ratio
        self.ch_in = ch_in; self.ch_out = ch_out
        self.ksize = ksize; self.stride = stride
        self.padding = padding; self.kernel = None
        self.gamma = nn.Parameter(torch.FloatTensor([2.0]))
        self.sigma = nn.Parameter(torch.FloatTensor([9.2 * r]))
        self.theta = nn.Parameter(
            torch.arange(ch_out).float() * math.pi / ch_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([0.057 / r]))
        self.psi = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=False)

    def _build_bank(self):
        xm  = self.ksize // 2
        rng = torch.arange(-xm, xm + 1).float()
        y   = rng.view(1, -1).repeat(self.ch_out, self.ch_in, self.ksize, 1)
        x   = rng.view(-1, 1).repeat(self.ch_out, self.ch_in, 1, self.ksize)
        x   = x.to(self.sigma.device); y = y.to(self.sigma.device)
        th  = self.theta.view(-1, 1, 1, 1)
        xt  =  x * torch.cos(th) + y * torch.sin(th)
        yt  = -x * torch.sin(th) + y * torch.cos(th)
        gb  = -torch.exp(
            -0.5 * ((self.gamma * xt) ** 2 + yt ** 2)
            / (8 * self.sigma.view(-1, 1, 1, 1) ** 2)
        ) * torch.cos(2 * math.pi * self.f.view(-1, 1, 1, 1) * xt
                      + self.psi.view(-1, 1, 1, 1))
        return gb - gb.mean(dim=[2, 3], keepdim=True)

    def forward(self, x):
        self.kernel = self._build_bank()
        return F.conv2d(x, self.kernel, stride=self.stride,
                        padding=self.padding)


class SELayer(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, ch, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class CompetitiveBlock(nn.Module):
    def __init__(self, ch_in, n_comp, ksize, weight,
                 init_ratio=1., o1=32):
        super().__init__()
        nc2 = n_comp * 2; nc4 = n_comp * 4
        self.g1 = GaborConv2d(ch_in, n_comp, ksize, 2, ksize // 2, init_ratio)
        self.g2 = GaborConv2d(nc2, nc2, ksize, 2, ksize // 2, init_ratio)
        if ksize == 35:
            self.c1a = nn.Conv2d(ch_in,  n_comp, 7, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 5, 2, 5)
            self.c2a = nn.Conv2d(nc2,    nc2,    7, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    5, 2, 5)
        elif ksize == 17:
            self.c1a = nn.Conv2d(ch_in,  n_comp, 5, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 3, 2, 3)
            self.c2a = nn.Conv2d(nc2,    nc2,    5, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    3, 2, 3)
        else:
            self.c1a = nn.Conv2d(ch_in,  n_comp, 3, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 1, 2, 1)
            self.c2a = nn.Conv2d(nc2,    nc2,    3, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    1, 2, 1)
        self.sm_c = nn.Softmax(dim=1)
        self.sm_h = nn.Softmax(dim=2)
        self.sm_w = nn.Softmax(dim=3)
        self.se1  = SELayer(nc2); self.se2 = SELayer(nc4)
        self.ppu1 = nn.Conv2d(nc2, o1 // 2, 5, 2, 0)
        self.ppu2 = nn.Conv2d(nc4, o1 // 2, 5, 2, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.wc   = weight; self.ws = (1. - weight) / 2.

    def _compete(self, x):
        return (self.wc * self.sm_c(x)
                + self.ws * (self.sm_h(x) + self.sm_w(x)))

    def forward(self, x):
        f  = torch.cat([self.g1(x), self.c1b(self.c1a(x))], dim=1)
        x1 = self.pool(self.ppu1(self.se1(self._compete(f))))
        f  = torch.cat([self.g2(f), self.c2b(self.c2a(f))], dim=1)
        x2 = self.pool(self.ppu2(self.se2(self._compete(f))))
        return torch.cat([x1.flatten(1), x2.flatten(1)], dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_f, out_f, s=ARC_S, m=ARC_M):
        super().__init__()
        self.s     = s
        self.w     = Parameter(torch.FloatTensor(out_f, in_f))
        nn.init.xavier_uniform_(self.w)
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cos = F.linear(F.normalize(x), F.normalize(self.w))
        if self.training and label is not None:
            sin = torch.sqrt((1. - cos ** 2).clamp(0., 1.))
            phi = cos * self.cos_m - sin * self.sin_m
            phi = torch.where(cos > self.th, phi, cos - self.mm)
            oh  = torch.zeros_like(cos).scatter_(
                1, label.view(-1, 1).long(), 1)
            return ((oh * phi) + ((1. - oh) * cos)) * self.s
        return self.s * cos

    def cosine_scores(self, x):
        return F.linear(F.normalize(x), F.normalize(self.w)) * self.s


class CCNet(nn.Module):
    """CCNet with 2-channel input [FAT | HM] — unchanged from PalmRSS."""
    def __init__(self, num_classes, weight=COM_WEIGHT):
        super().__init__()
        self.cb1  = CompetitiveBlock(2,  9, 35, weight, init_ratio=1.00)
        self.cb2  = CompetitiveBlock(2, 36, 17, weight, init_ratio=0.50)
        self.cb3  = CompetitiveBlock(2,  9,  7, weight, init_ratio=0.25)
        self.fc   = nn.Linear(13152, FC_DIM1)
        self.fc1  = nn.Linear(FC_DIM1, FC_DIM2)
        self.drop = nn.Dropout(DROPOUT)
        self.arc  = ArcMarginProduct(FC_DIM2, num_classes, s=ARC_S, m=ARC_M)

    def _backbone(self, x):
        return torch.cat([self.cb1(x), self.cb2(x), self.cb3(x)], dim=1)

    def forward(self, x, y=None):
        h1  = self.fc(self._backbone(x))
        h2  = self.fc1(h1)
        fe  = torch.cat([h1, h2], dim=1)
        out = self.arc(self.drop(h2), y)
        return out, F.normalize(fe, dim=-1)

    def cosine_classify(self, x):
        h2 = self.fc1(self.fc(self._backbone(x)))
        return self.arc.cosine_scores(self.drop(h2))

    def getFeatureCode(self, x):
        return F.normalize(self.fc1(self.fc(self._backbone(x))), dim=-1)


# ============================================================
# DOMAIN DISCRIMINATOR  (unchanged from original PalmRSS)
# ============================================================

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=FC_DIM1 + FC_DIM2, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.net(x)


# ============================================================
# LOSSES  (unchanged from original PalmRSS)
# ============================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE, base_temperature=BASE_TEMP):
        super().__init__()
        self.T = temperature; self.base = base_temperature

    def forward(self, features, labels):
        dev = features.device
        bsz = features.shape[0]; n = features.shape[1]
        mask = torch.eq(
            labels.view(-1, 1), labels.view(1, -1)).float().to(dev)
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        dot      = torch.div(torch.matmul(contrast, contrast.T), self.T)
        lm, _    = torch.max(dot, dim=1, keepdim=True)
        logits   = dot - lm.detach()
        mask     = mask.repeat(n, n)
        lmask    = 1. - torch.eye(bsz * n, device=dev)
        mask     = mask * lmask
        exp_log  = torch.exp(logits) * lmask
        log_prob = logits - torch.log(exp_log.sum(1, keepdim=True) + 1e-9)
        denom    = mask.sum(1).clamp(min=1.)
        return (-(self.T / self.base) * (mask * log_prob).sum(1) / denom).mean()


def feature_similarity_loss(v, v_aug):
    return (1. - F.cosine_similarity(v, v_aug, dim=-1)).mean()


def adversarial_loss(disc, v1, v2):
    bce  = nn.BCEWithLogitsLoss()
    lbl1 = torch.ones (v1.size(0), 1, device=v1.device)
    lbl2 = torch.zeros(v2.size(0), 1, device=v2.device)
    return bce(disc(v1), lbl1) + bce(disc(v2), lbl2)


# ============================================================
# IMAGE ALIGNMENT  (unchanged from original PalmRSS)
# ============================================================

def _hist_match_np(src, tgt):
    matched = np.empty_like(src)
    for c in range(src.shape[2]):
        s = src[..., c].ravel().astype(np.float64)
        t = tgt[..., c].ravel().astype(np.float64)
        s_min, s_max = s.min(), s.max()
        t_min, t_max = t.min(), t.max()
        if s_max == s_min or t_max == t_min:
            matched[..., c] = src[..., c]; continue
        s_n = (s - s_min) / (s_max - s_min)
        t_n = (t - t_min) / (t_max - t_min)
        bins  = 256
        s_cnt, _ = np.histogram(s_n, bins=bins, range=(0., 1.))
        t_cnt, _ = np.histogram(t_n, bins=bins, range=(0., 1.))
        s_cdf = np.cumsum(s_cnt).astype(np.float64); s_cdf /= s_cdf[-1]
        t_cdf = np.cumsum(t_cnt).astype(np.float64); t_cdf /= t_cdf[-1]
        edges   = np.linspace(0., 1., bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.
        t_idx   = np.searchsorted(t_cdf, s_cdf).clip(0, bins - 1)
        lut     = centers[t_idx] * (s_max - s_min) + s_min
        pix_bin = np.searchsorted(edges[1:], s_n).clip(0, bins - 1)
        matched[..., c] = lut[pix_bin].reshape(src.shape[:2]).astype(np.float32)
    return matched.astype(np.float32)


def hm_batch(src_batch, tgt_batch):
    rows = []
    for s, t in zip(src_batch, tgt_batch):
        s_np = s.permute(1, 2, 0).numpy()
        t_np = t.permute(1, 2, 0).numpy()
        rows.append(
            torch.from_numpy(_hist_match_np(s_np, t_np)).permute(2, 0, 1))
    return torch.stack(rows).float()


def fat_batch(src, tgt, beta=BETA):
    fs  = torch.fft.rfft2(src, dim=(-2, -1))
    ft  = torch.fft.rfft2(tgt, dim=(-2, -1))
    as_ = torch.abs(fs).clone(); ps = torch.angle(fs); at = torch.abs(ft)
    _, _, h, w2 = as_.shape
    bh = int(np.floor(beta * h)); bw = int(np.floor(beta * w2 * 2))
    b  = min(bh, bw)
    if b > 0:
        as_[:, :, :b,      :b] = at[:, :, :b,      :b]
        as_[:, :, h-b+1:h, :b] = at[:, :, h-b+1:h, :b]
    rec = torch.fft.irfft2(
        torch.complex(torch.cos(ps) * as_, torch.sin(ps) * as_),
        dim=(-2, -1), s=[h, w2 * 2])
    return rec[..., :src.shape[-2], :src.shape[-1]]


def make_2ch(src, tgt):
    return torch.cat([fat_batch(src, tgt), hm_batch(src, tgt)], dim=1)


def make_2ch_identity(x):
    return torch.cat([x, x], dim=1)


# ============================================================
# EVALUATION  (gallery vs probe → EER + Rank-1)
# ============================================================

def compute_eer(ins, outs):
    if ins.mean() < outs.mean():
        ins, outs = -ins, -outs
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    sc  = np.concatenate([ins, outs])
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100.


@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, ids = [], []
    for (d, _), target in loader:
        codes = model.getFeatureCode(make_2ch_identity(d).to(DEVICE))
        feats.append(codes.cpu().numpy())
        ids.append(target.numpy())
    return np.concatenate(feats), np.concatenate(ids)


def evaluate(model, gallery_loader, probe_loader, out_dir=".", tag="eval"):
    """Cosine similarity gallery-probe evaluation → EER + Rank-1."""
    ft_g, id_g = extract_features(model, gallery_loader)
    ft_p, id_p = extract_features(model, probe_loader)

    sim   = ft_p @ ft_g.T
    rank1 = 100. * (id_g[sim.argmax(axis=1)] == id_p).mean()

    dis   = np.arccos(np.clip(sim, -1., 1.)) / np.pi
    n_g, n_p = len(id_g), len(id_p)
    gal_ids_tiled = np.tile(id_g, n_p)
    probe_ids_rep = np.repeat(id_p, n_g)
    l   = np.where(gal_ids_tiled == probe_ids_rep, 1, -1)
    s   = dis.ravel()
    ins  = 1. - s[l ==  1]
    outs = 1. - s[l == -1]
    eer  = compute_eer(ins, outs)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for score, label in zip(s, l):
            f.write(f"{1.-score:.6f} {label}\n")

    log(f"  [{tag}]  EER={eer:.4f}%  Rank-1={rank1:.3f}%")
    return eer, rank1


# ============================================================
# TRAINING LOOP  (unchanged from original PalmRSS)
# ============================================================

def fit_epoch(epoch, model, disc, d1_loader, d2_iter_ref,
              criterion, con_crit, opt_model, opt_disc):
    model.train(); disc.train()
    run_loss = 0.; arc_corr = 0; cos_corr = 0; total = 0

    for (x_d1, x_d1_aug), y_d1 in d1_loader:
        try:
            (x_d2, _), _ = next(d2_iter_ref[0])
        except StopIteration:
            d2_iter_ref[0] = iter(d2_iter_ref[1])
            (x_d2, _), _  = next(d2_iter_ref[0])

        y_d1 = y_d1.to(DEVICE)
        data     = make_2ch(x_d1,     x_d2).to(DEVICE)
        data_aug = make_2ch(x_d1_aug, x_d2).to(DEVICE)
        data_d2  = make_2ch(x_d2,     x_d1).to(DEVICE)

        opt_model.zero_grad(); opt_disc.zero_grad()

        out1, fe1 = model(data,     y_d1)
        out2, fe2 = model(data_aug, y_d1)

        with torch.no_grad():
            _, fe_d2 = model(data_d2, None)

        l_ce  = criterion(out1, y_d1)
        l_con = con_crit(torch.stack([fe1, fe2], dim=1), y_d1)
        l_sim = feature_similarity_loss(fe1, fe2)
        l_hyb = W_CE * l_ce + W_CON * l_con + W_SIM * l_sim
        l_adv = adversarial_loss(disc, fe1.detach(), fe_d2.detach())
        loss  = l_adv + LAMBDA_HYB * l_hyb

        loss.backward()
        opt_model.step(); opt_disc.step()

        run_loss += loss.item() * y_d1.size(0)
        total    += y_d1.size(0)

        with torch.no_grad():
            arc_corr += out1.argmax(1).eq(y_d1).sum().item()
            model.eval()
            cos_corr += (model.cosine_classify(data)
                         .argmax(1).eq(y_d1).sum().item())
            model.train(); disc.train()

    return (run_loss / total,
            100. * arc_corr / total,
            100. * cos_corr / total)


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(d1_samples, d2_samples, gallery_samples, probe_samples,
                   num_classes, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    kw = dict(num_workers=4, pin_memory=True)
    d1_ds = PalmDataset(d1_samples, train=True)
    d2_ds = PalmDataset(d2_samples, train=True)

    d1_loader  = DataLoader(d1_ds, batch_size=min(BATCH_SIZE, len(d1_ds)),
                            shuffle=True, drop_last=True, **kw)
    d2_loader  = DataLoader(d2_ds, batch_size=min(BATCH_SIZE, len(d2_ds)),
                            shuffle=True, drop_last=True, **kw)
    gal_loader = DataLoader(
        PalmDataset(gallery_samples, train=False),
        batch_size=min(BATCH_SIZE, len(gallery_samples)), shuffle=False, **kw)
    prb_loader = DataLoader(
        PalmDataset(probe_samples, train=False),
        batch_size=min(BATCH_SIZE, len(probe_samples)), shuffle=False, **kw)

    net  = CCNet(num_classes, COM_WEIGHT).to(DEVICE)
    disc = DomainDiscriminator(FC_DIM1 + FC_DIM2, 1024).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    con_crit  = SupConLoss(TEMPERATURE, BASE_TEMP)
    opt_model = optim.Adam(net.parameters(),  lr=LR)
    opt_disc  = optim.Adam(disc.parameters(), lr=LR)
    sched     = lr_scheduler.StepLR(opt_model, step_size=LR_STEP, gamma=LR_GAMMA)

    best_rank1 = 0.0
    ckpt_path  = os.path.join(results_dir, "best_model.pth")
    d2_iter_ref = [iter(d2_loader), d2_loader]

    log(f"  Loss: L_adv + {LAMBDA_HYB}×({W_CE}·L_ce + {W_CON}·L_con + {W_SIM}·L_sim)")
    log(f"  D1={len(d1_samples)}  D2={len(d2_samples)}  "
        f"Gallery={len(gallery_samples)}  Probe={len(probe_samples)}")

    # Pre-training baseline
    evaluate(net, gal_loader, prb_loader, rst_eval, "ep000_pretrain")

    for epoch in range(EPOCH_NUM):
        loss, arc_acc, cos_acc = fit_epoch(
            epoch, net, disc, d1_loader, d2_iter_ref,
            criterion, con_crit, opt_model, opt_disc)
        sched.step()

        if epoch % PRINT_INTERVAL == 0 or epoch == EPOCH_NUM - 1:
            cur_eer, cur_rank1 = evaluate(
                net, gal_loader, prb_loader,
                rst_eval, f"ep{epoch:04d}")
            marker = "  *** new best ***" if cur_rank1 > best_rank1 else ""
            log(f"  ep {epoch:03d}/{EPOCH_NUM}  "
                f"loss={loss:.4f}  arc={arc_acc:.1f}%  cos={cos_acc:.1f}%  "
                f"EER={cur_eer:.4f}%  Rank-1={cur_rank1:.2f}%{marker}")
            if cur_rank1 > best_rank1:
                best_rank1 = cur_rank1
                torch.save({"epoch": epoch, "model": net.state_dict(),
                            "rank1": cur_rank1, "eer": cur_eer}, ckpt_path)
        else:
            if epoch % (PRINT_INTERVAL // 2) == 0:
                log(f"  ep {epoch:03d}/{EPOCH_NUM}  "
                    f"loss={loss:.4f}  arc={arc_acc:.1f}%  cos={cos_acc:.1f}%")

    # Reload best checkpoint for final result
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    net.load_state_dict(ckpt["model"])
    final_eer, final_rank1 = evaluate(
        net, gal_loader, prb_loader, rst_eval, "FINAL")
    log(f"  Best Rank-1={best_rank1:.2f}%  Final: EER={final_eer:.4f}%  Rank-1={final_rank1:.2f}%")

    return final_eer, final_rank1


# ============================================================
# RESULTS SUMMARY TABLE
# ============================================================

def print_and_save_summary(all_results, out_path):
    col_w  = 14
    header = (f"{'Setting':<22}"
              f"{'Train domain':<38}"
              f"{'Test domain':<26}"
              f"{'EER (%)':>{col_w}}"
              f"{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)
    lines = ["\nCross-Domain Closed-Set Results — Palm-Auth (PalmRSS)",
             sep, header, sep]
    for r in all_results:
        eer_str   = f"{r['eer']:.2f}"   if r['eer']   is not None else "—"
        rank1_str = f"{r['rank1']:.2f}" if r['rank1'] is not None else "—"
        lines.append(f"{r['setting']:<22}"
                     f"{r['train_desc']:<38}"
                     f"{r['test_desc']:<26}"
                     f"{eer_str:>{col_w}}"
                     f"{rank1_str:>{col_w}}")
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    log(f"Summary saved to: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    log("=" * 72)
    log(f"PalmRSS — Cross-Domain Closed-Set (Palm-Auth)")
    log(f"Device    : {DEVICE}")
    log(f"Epochs    : {EPOCH_NUM}")
    log(f"Settings  : 2 scanner + {len(PAIRED_CONDITIONS)} paired-condition")
    log(f"D1/D2     : 50/50 sample split per identity within training domain")
    log(f"Results   : {SAVE_DIR}")
    log("=" * 72)

    log("\nScanning dataset …")
    cond_paths    = _collect_perspective(PALM_AUTH_ROOT)
    scanner_paths = _collect_scanner(PALM_AUTH_ROOT, SCANNER_SPECTRA)
    log(f"Perspective conditions: {sorted(cond_paths.keys())}")
    log(f"Scanner identities   : {len(scanner_paths)}")

    all_splits = load_or_generate_closedset_splits(
        cond_paths, scanner_paths, TEST_GALLERY_RATIO, SEED)

    SETTINGS = []

    SETTINGS.append({
        "tag"        : "setting_scanner",
        "label"      : "S_scanner",
        "train_desc" : "Perspective (all 190 IDs)",
        "test_desc"  : "Scanner 50/50 gallery/probe",
        "parser"     : lambda: parse_setting_scanner(
                           cond_paths, scanner_paths, all_splits, SEED),
    })
    SETTINGS.append({
        "tag"        : "setting_scanner_to_persp",
        "label"      : "S_scanner_to_persp",
        "train_desc" : "Scanner (148 IDs)",
        "test_desc"  : "Perspective 50/50 gallery/probe",
        "parser"     : lambda: parse_setting_scanner_to_perspective(
                           cond_paths, scanner_paths, all_splits, SEED),
    })

    conditions_found = sorted(cond_paths.keys())
    for cond_a, cond_b in PAIRED_CONDITIONS:
        if cond_a not in conditions_found or cond_b not in conditions_found:
            log(f"  [WARN] '{cond_a}' or '{cond_b}' not found — skipping")
            continue
        ca, cb = cond_a, cond_b
        SETTINGS.append({
            "tag"        : f"setting_{ca}_{cb}",
            "label"      : f"S_{ca}_{cb}",
            "train_desc" : f"Perspective(not {ca}/{cb}) + Scanner",
            "test_desc"  : f"gallery:{ca} / probe:{cb}",
            "parser"     : (lambda ca=ca, cb=cb: parse_setting_paired_conditions(
                                ca, cb, cond_paths, scanner_paths, SEED)),
        })

    log(f"\nTotal settings to run : {len(SETTINGS)}")

    all_results = []

    for idx, s in enumerate(SETTINGS, 1):
        log(f"\n{'='*72}")
        log(f"[{idx}/{len(SETTINGS)}] {s['label']}")
        log(f"  Train : {s['train_desc']}")
        log(f"  Test  : {s['test_desc']}")
        log(f"{'='*72}")

        results_dir = os.path.join(SAVE_DIR, s["tag"])
        t_start     = time.time()
        try:
            d1_s, d2_s, gal_s, prb_s, n_cls = s["parser"]()
            eer, rank1 = run_experiment(
                d1_s, d2_s, gal_s, prb_s, n_cls, results_dir)
            elapsed = time.time() - t_start
            log(f"\n  ✓  {s['label']}:  EER={eer:.4f}%  "
                f"Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            with open(os.path.join(results_dir, "results.json"), "w") as f:
                json.dump({"setting": s["label"], "train_desc": s["train_desc"],
                           "test_desc": s["test_desc"], "num_classes": n_cls,
                           "EER_pct": eer, "Rank1_pct": rank1}, f, indent=2)
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": eer, "rank1": rank1})
        except Exception as e:
            log(f"\n  ✗  {s['label']} FAILED: {e}")
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": None, "rank1": None})

    log(f"\n\n{'='*72}")
    log(f"ALL {len(SETTINGS)} SETTINGS COMPLETE")
    log(f"{'='*72}")
    print_and_save_summary(
        all_results,
        os.path.join(SAVE_DIR, "results_summary.txt"))


if __name__ == "__main__":
    main()
