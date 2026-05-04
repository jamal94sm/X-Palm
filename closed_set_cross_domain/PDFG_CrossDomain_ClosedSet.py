"""
PDFG: Palm-print Domain Feature Generalization
Adapted for Palm-Auth dataset with cross-domain closed-set evaluation.

Model, training method, hyperparameters: unchanged from original PDFG.
Dataset      : Palm-Auth (roi_perspective + roi_scanner)
Images       : loaded as original RGB (convert("RGB")) — SharedLayers expects 3-ch
Evaluation   : Cross-Domain Closed-Set (12 settings)
               gallery vs probe → EER + Rank-1
Checkpoint   : saved by best Rank-1

Fourier augmentation:
  Within each training batch, a random permutation of batch indices provides
  the "style" images for Fourier augmentation — no D1/D2 split needed.
  N=1 head. At eval, extract_avg() → same as extract(x, 0) for N=1.

Settings (12 total)
────────────────────
  S_scanner         │ Train : perspective (all, 190 IDs)
                    │ Gallery: 50% scanner (148 IDs)
                    │ Probe  : 50% scanner (148 IDs)

  S_scanner_to_persp│ Train : scanner (148 IDs)
                    │ Gallery: 50% perspective (148 IDs)
                    │ Probe  : 50% perspective (148 IDs)

  S_(A,B) (×10)     │ Train : perspective(¬A,¬B) + scanner
                    │ Gallery: ALL condition A images
                    │ Probe  : ALL condition B images

Gallery/probe splits saved to palm_auth_closedset_splits.json on first run.
"""

import json
import os
import random
import time
import warnings
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_metric_learning import losses as pml_losses

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (unchanged from original PDFG)
# ─────────────────────────────────────────────────────────────────────────────

PALM_AUTH_ROOT  = "/home/pai-ng/Jamal/smartphone_data"
SCANNER_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

TEST_GALLERY_RATIO = 0.50
SPLITS_FILE        = "./palm_auth_closedset_splits.json"
SAVE_DIR           = "./rst_pdfg_crossdomain"

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

FEATURE_DIM     = 128
ARCFACE_S       = 32.0
ARCFACE_M       = 0.35
TRIPLET_MARGIN  = 0.4
ALPHA           = 0.1
BETA            = 1.0
LAM             = 0.8

BATCH_SIZE      = 32
LR              = 1e-4
EPOCHS          = 200
PRETRAIN_EPOCHS = 20
EVAL_EVERY      = 5

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}

# N=2 heads: D1 and D2 are condition-based splits (same conditions for all IDs)
N = 2

# Conditions assigned to D1 and D2 — same split applied to all settings.
# For scanner-only training (S_scanner_to_persp), scanner spectra are split
# between D1 (D1_SCANNER_SPECTRA) and D2 (the rest).
D1_CONDITIONS      = ["bf", "close", "far", "fl", "jf", "roll"]
D2_CONDITIONS      = ["pitch", "rnd", "sf", "text", "wet"]
D1_SCANNER_SPECTRA = {"white", "ir"}      # scanner spectra → D1
D2_SCANNER_SPECTRA = {"yellow", "pink", "green"}  # scanner spectra → D2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOURIER AUGMENTATION  (unchanged from original PDFG)
# ─────────────────────────────────────────────────────────────────────────────

def fourier_augment_batch(batch1, batch2, lam=LAM):
    fft1 = torch.fft.fft2(batch1, dim=(-2, -1))
    fft2 = torch.fft.fft2(batch2, dim=(-2, -1))
    amp1, phase1 = torch.abs(fft1), torch.angle(fft1)
    amp2      = torch.abs(fft2)
    amp_mixed = (1 - lam) * amp1 + lam * amp2
    fft_new   = amp_mixed * torch.exp(1j * phase1)
    result    = torch.real(torch.fft.ifft2(fft_new, dim=(-2, -1)))
    return torch.clamp(result, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_perspective(data_root):
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


def _condition_split_d1_d2(cond_paths, scanner_paths, label_map, eligible_ids,
                            exclude_conds=()):
    """
    Condition-based D1/D2 split: same conditions for all IDs.
      D1 = images from D1_CONDITIONS (minus excluded test conditions)
           + scanner images from D1_SCANNER_SPECTRA
      D2 = images from D2_CONDITIONS (minus excluded test conditions)
           + scanner images from D2_SCANNER_SPECTRA
    """
    d1, d2 = [], []
    for ident in eligible_ids:
        lbl = label_map[ident]
        for cond in D1_CONDITIONS:
            if cond in exclude_conds: continue
            for p in cond_paths.get(cond, {}).get(ident, []):
                d1.append((p, lbl))
        for cond in D2_CONDITIONS:
            if cond in exclude_conds: continue
            for p in cond_paths.get(cond, {}).get(ident, []):
                d2.append((p, lbl))
        for p in scanner_paths.get(ident, []):
            # Split scanner by spectra
            fname = os.path.basename(p)
            parts = os.path.splitext(fname)[0].split("_")
            spec  = parts[2].lower() if len(parts) >= 3 else ""
            if spec in D1_SCANNER_SPECTRA:
                d1.append((p, lbl))
            else:
                d2.append((p, lbl))
    return d1, d2


def _scanner_split_d1_d2(scanner_paths, label_map, eligible_ids):
    """
    For scanner-only training (S_scanner_to_persp):
    Split scanner images by spectra into D1 and D2.
    """
    d1, d2 = [], []
    for ident in eligible_ids:
        lbl = label_map[ident]
        for p in scanner_paths.get(ident, []):
            fname = os.path.basename(p)
            parts = os.path.splitext(fname)[0].split("_")
            spec  = parts[2].lower() if len(parts) >= 3 else ""
            if spec in D1_SCANNER_SPECTRA:
                d1.append((p, lbl))
            else:
                d2.append((p, lbl))
    return d1, d2



# ─────────────────────────────────────────────────────────────────────────────
# GALLERY/PROBE SPLIT PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

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
    return {
        "S_scanner": _make_gallery_probe_split(
            {i: scanner_paths[i] for i in scanner_ids}, gallery_ratio, rng),
        "S_scanner_to_persp": _make_gallery_probe_split(
            {i: persp_all[i] for i in scanner_ids}, gallery_ratio, rng),
    }


def load_or_generate_closedset_splits(cond_paths, scanner_paths, gallery_ratio, seed):
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f: splits = json.load(f)
        log(f"Loaded existing gallery/probe splits from: {SPLITS_FILE}")
    else:
        log(f"Generating gallery/probe splits (seed={seed}) → {SPLITS_FILE}")
        splits = generate_closedset_splits(cond_paths, scanner_paths, gallery_ratio, seed)
        with open(SPLITS_FILE, "w") as f: json.dump(splits, f, indent=2)
        log(f"Splits saved to: {SPLITS_FILE}")
    for key, val in splits.items():
        n_gal = sum(len(v["gallery"]) for v in val.values())
        n_prb = sum(len(v["probe"])   for v in val.values())
        log(f"  {key:<30}  IDs={len(val)}  gallery={n_gal}  probe={n_prb}")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS — return (d1, d2, gallery, probe, num_classes)
# ─────────────────────────────────────────────────────────────────────────────

def parse_setting_scanner(cond_paths, scanner_paths, stored_splits, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    all_persp_ids   = sorted(persp_all.keys())
    scanner_ids     = sorted(scanner_paths.keys())
    train_label_map = {ident: i for i, ident in enumerate(all_persp_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    d1, d2 = _condition_split_d1_d2(
        cond_paths, {}, train_label_map, all_persp_ids, exclude_conds=())
    gallery, probe = _gallery_probe_split_from_stored(
        {i: scanner_paths[i] for i in scanner_ids}, test_label_map,
        stored_splits["S_scanner"])
    _print_stats("S_scanner", len(all_persp_ids), len(scanner_ids),
                 len(d1), len(d2), len(gallery), len(probe))
    return d1, d2, gallery, probe, len(all_persp_ids)


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths, stored_splits, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    scanner_ids     = sorted(scanner_paths.keys())
    train_label_map = {ident: i for i, ident in enumerate(scanner_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    d1, d2 = _scanner_split_d1_d2(scanner_paths, train_label_map, scanner_ids)
    gallery, probe = _gallery_probe_split_from_stored(
        {i: persp_all[i] for i in scanner_ids}, test_label_map,
        stored_splits["S_scanner_to_persp"])
    _print_stats("S_scanner_to_persp", len(scanner_ids), len(scanner_ids),
                 len(d1), len(d2), len(gallery), len(probe))
    return d1, d2, gallery, probe, len(scanner_ids)


def parse_setting_paired_conditions(cond_a, cond_b, cond_paths, scanner_paths, seed):
    paths_a = cond_paths.get(cond_a, {}); paths_b = cond_paths.get(cond_b, {})
    if not paths_a: raise ValueError(f"No images for condition '{cond_a}'")
    if not paths_b: raise ValueError(f"No images for condition '{cond_b}'")
    eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
    if not eligible_ids: raise ValueError(f"No IDs with both '{cond_a}' and '{cond_b}'")
    label_map    = {ident: i for i, ident in enumerate(eligible_ids)}
    train_id2paths = defaultdict(list)
    for cond, cond_dict in cond_paths.items():
        if cond in (cond_a, cond_b): continue
        for ident in eligible_ids:
            train_id2paths[ident].extend(cond_dict.get(ident, []))
    for ident in eligible_ids:
        train_id2paths[ident].extend(scanner_paths.get(ident, []))
    # D1/D2 condition-based split — exclude test conditions from both groups
    d1, d2 = _condition_split_d1_d2(
        cond_paths, scanner_paths, label_map, eligible_ids,
        exclude_conds=(cond_a, cond_b))
    gallery = _all_samples({i: paths_a[i] for i in eligible_ids}, label_map)
    probe   = _all_samples({i: paths_b[i] for i in eligible_ids}, label_map)
    _print_stats(f"S_{cond_a}_{cond_b}", len(eligible_ids), len(eligible_ids),
                 len(d1), len(d2), len(gallery), len(probe))
    return d1, d2, gallery, probe, len(eligible_ids)


def _print_stats(name, n_train, n_test, d1_n, d2_n, gallery_n, probe_n):
    log(f"  [{name}]")
    log(f"    Train IDs / Test IDs : {n_train} / {n_test}")
    log(f"    D1 ({','.join(D1_CONDITIONS)}) : {d1_n}")
    log(f"    D2 ({','.join(D2_CONDITIONS)}) : {d2_n}")
    log(f"    Gallery / Probe      : {gallery_n} / {probe_n}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# Palm-Auth images are RGB — SharedLayers has Conv2d(3, ...) input
# ─────────────────────────────────────────────────────────────────────────────

class PalmAuthDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.tf = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),  # [0,1] RGB — no normalisation (original PDFG)
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.tf(Image.open(path).convert("RGB")), label


def make_loader(samples, batch_size=BATCH_SIZE, shuffle=False, drop_last=False):
    ds = PalmAuthDataset(samples)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)),
                      shuffle=shuffle, num_workers=4, pin_memory=True,
                      drop_last=drop_last and len(ds) > batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (unchanged from original PDFG)
# ─────────────────────────────────────────────────────────────────────────────

class SharedLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        self.conv3 = nn.Conv2d(32, 64,  3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = self.pool3(self.act(self.conv4(x)))
        return x


class MultiDatasetExtractors(nn.Module):
    def __init__(self, n_datasets, feature_dim=FEATURE_DIM):
        super().__init__()
        self.n_datasets = n_datasets
        self.shared     = SharedLayers()
        with torch.no_grad():
            flat_dim = self.shared(torch.zeros(1, 3, 112, 112)).view(1, -1).shape[1]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 1024), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 512),      nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, feature_dim),
            ) for _ in range(n_datasets)
        ])

    def extract(self, x, idx):
        f = self.heads[idx](self.shared(x).view(x.size(0), -1))
        return F.normalize(f, p=2, dim=1)

    def extract_all(self, x):
        shared = self.shared(x).view(x.size(0), -1)
        return [F.normalize(h(shared), p=2, dim=1) for h in self.heads]

    def extract_avg(self, x):
        """Paper eval: average all N heads → L2-normalise."""
        per_head = torch.stack(self.extract_all(x), dim=0)
        avg      = per_head.mean(dim=0)
        return F.normalize(avg, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# LOSSES  (unchanged from original PDFG)
# ─────────────────────────────────────────────────────────────────────────────

def mkmmd_loss(f1, f2, kernels=(1, 5, 10, 20, 50, 100)):
    def sq_dists(a, b):
        aa = (a * a).sum(dim=1, keepdim=True)
        bb = (b * b).sum(dim=1, keepdim=True)
        return (aa + bb.t() - 2 * torch.mm(a, b.t())).clamp(min=0)
    d_ss = sq_dists(f1, f1); d_st = sq_dists(f1, f2); d_tt = sq_dists(f2, f2)
    loss = 0.0
    for bw in kernels:
        loss += (torch.exp(-d_ss/bw).mean()
                 - 2*torch.exp(-d_st/bw).mean()
                 + torch.exp(-d_tt/bw).mean())
    return loss / len(kernels)


def consistent_loss(orig_feat, aug_feats_per_pair):
    loss = torch.tensor(0.0, device=orig_feat.device)
    for head_feats in aug_feats_per_pair:
        avg   = torch.stack(head_feats, dim=0).mean(0)
        loss += ((orig_feat - avg) ** 2).sum(dim=1).mean()
    return loss


def triplet_loss_fn(anchor, positive, negative, margin=TRIPLET_MARGIN):
    return F.relu(
        F.pairwise_distance(anchor, positive) -
        F.pairwise_distance(anchor, negative) + margin
    ).mean()


def sample_triplet_pairs(aug_avg, aug_labels, anchor_labels):
    B = anchor_labels.size(0)
    positives = aug_avg.clone()
    negatives = torch.zeros_like(aug_avg)
    for i in range(B):
        pool = (aug_labels != anchor_labels[i]).nonzero(as_tuple=False).squeeze(1)
        negatives[i] = (aug_avg[pool[torch.randint(len(pool), (1,))]]
                        if len(pool) > 0 else aug_avg[random.randint(0, B-1)])
    return positives, negatives


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  (gallery vs probe → EER + Rank-1)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features_avg(model, loader):
    model.eval()
    feats, labels = [], []
    for imgs, lbl in loader:
        feats.append(model.extract_avg(imgs.to(DEVICE)).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer_metric(scores_array):
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return 1.0, 0.0
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, thresh


def evaluate(model, gallery_loader, probe_loader, out_dir=".", tag="eval"):
    ft_g, id_g = extract_features_avg(model, gallery_loader)
    ft_p, id_p = extract_features_avg(model, probe_loader)
    sim   = ft_p @ ft_g.T
    rank1 = 100.0 * (id_g[sim.argmax(axis=1)] == id_p).mean()
    scores_list, labels_list = [], []
    for i in range(len(id_p)):
        for j in range(len(id_g)):
            scores_list.append(float(sim[i, j]))
            labels_list.append(1 if id_p[i] == id_g[j] else -1)
    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer_metric(scores_arr)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")
    log(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


# ─────────────────────────────────────────────────────────────────────────────
# INFINITE LOADER HELPER  (unchanged from original PDFG)
# ─────────────────────────────────────────────────────────────────────────────

class _Inf:
    def __init__(self, loader):
        self.loader = loader; self._it = iter(loader)
    def next(self):
        try: return next(self._it)
        except StopIteration:
            self._it = iter(self.loader); return next(self._it)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(d1_samples, d2_samples, gallery_samples, probe_samples,
                   num_classes, results_dir):
    """
    N=2 heads trained on condition-based D1 and D2.
    Fourier augmentation mixes images from D1 with style from D2 and vice versa.
    """
    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval"); os.makedirs(rst_eval, exist_ok=True)

    d1_loader  = make_loader(d1_samples, BATCH_SIZE, shuffle=True, drop_last=True)
    d2_loader  = make_loader(d2_samples, BATCH_SIZE, shuffle=True, drop_last=True)
    gal_loader = make_loader(gallery_samples, 128)
    prb_loader = make_loader(probe_samples,   128)

    model     = MultiDatasetExtractors(N, FEATURE_DIM).to(DEVICE)
    arc_heads = nn.ModuleList([
        pml_losses.ArcFaceLoss(num_classes=num_classes, embedding_size=FEATURE_DIM,
                               margin=ARCFACE_M, scale=ARCFACE_S).to(DEVICE)
        for _ in range(N)
    ])
    all_params      = list(model.parameters()) + list(arc_heads.parameters())
    optimizer       = optim.Adam(all_params, lr=LR)
    train_loaders   = [d1_loader, d2_loader]
    inf_loaders     = [_Inf(ld) for ld in train_loaders]
    steps_per_epoch = min(len(ld) for ld in train_loaders)

    best_rank1 = 0.0
    ckpt_path  = os.path.join(results_dir, "best_model.pth")

    log(f"  N={N} heads | D1={len(d1_samples)} D2={len(d2_samples)} "
        f"steps/ep={steps_per_epoch} classes={num_classes}")
    log(f"  Gallery={len(gallery_samples)}  Probe={len(probe_samples)}")

    evaluate(model, gal_loader, prb_loader, rst_eval, "ep000_pretrain")

    # ── Phase 1: Supervised pre-training (L_sup only) ──────────────────────
    log(f"\n  Phase 1 — Supervised Pre-training ({PRETRAIN_EPOCHS} epochs)")
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        for h in arc_heads: h.train()
        ep_loss = 0.0; ep_corr = 0; ep_total = 0
        for _ in range(steps_per_epoch):
            batches = [(imgs.to(DEVICE), lbl.to(DEVICE))
                       for imgs, lbl in (il.next() for il in inf_loaders)]
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=DEVICE)
            for i, (src_imgs, src_lbl) in enumerate(batches):
                feat  = model.extract(src_imgs, i)
                loss += arc_heads[i](feat, src_lbl)
                with torch.no_grad():
                    ep_corr  += arc_heads[i].get_logits(feat).argmax(1).eq(src_lbl).sum().item()
                    ep_total += src_lbl.size(0)
                for j in range(N):
                    if i == j: continue
                    sty, _ = batches[j]
                    if sty.size(0) != src_imgs.size(0):
                        sty = sty[torch.randint(sty.size(0), (src_imgs.size(0),), device=DEVICE)]
                    loss += arc_heads[i](
                        model.extract(fourier_augment_batch(src_imgs, sty, LAM), i), src_lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()
            ep_loss += loss.item()
        avg_acc = 100.0 * ep_corr / ep_total if ep_total > 0 else 0.0
        log(f"  P1 ep {epoch+1:03d}/{PRETRAIN_EPOCHS}  "
            f"loss={ep_loss/steps_per_epoch:.4f}  acc={avg_acc:.2f}%")
        if (epoch + 1) % EVAL_EVERY == 0:
            cur_eer, cur_rank1 = evaluate(
                model, gal_loader, prb_loader, rst_eval, f"P1_ep{epoch+1:04d}")
            if cur_rank1 > best_rank1:
                best_rank1 = cur_rank1
                torch.save({"epoch": epoch+1, "model": model.state_dict(),
                            "arc_heads": [h.state_dict() for h in arc_heads],
                            "rank1": cur_rank1, "eer": cur_eer}, ckpt_path)
                log(f"  *** New best Rank-1: {best_rank1:.2f}% ***")

    # ── Phase 2: Full PDFG training (all losses) ───────────────────────────
    log(f"\n  Phase 2 — Full PDFG Training ({EPOCHS} epochs)")
    log(f"  L = L_sup + L_ada + {ALPHA}·L_con + {BETA}·L_d-t")
    for epoch in range(EPOCHS):
        model.train()
        for h in arc_heads: h.train()
        ll = {"total": 0., "sup": 0., "ada": 0., "con": 0., "dt": 0.}
        ep_corr = 0; ep_total = 0
        for _ in range(steps_per_epoch):
            batches = [(imgs.to(DEVICE), lbl.to(DEVICE))
                       for imgs, lbl in (il.next() for il in inf_loaders)]
            aug = {}
            for i in range(N):
                src_imgs, src_lbl = batches[i]
                for j in range(N):
                    if i == j: continue
                    sty, _ = batches[j]
                    if sty.size(0) != src_imgs.size(0):
                        sty = sty[torch.randint(sty.size(0), (src_imgs.size(0),), device=DEVICE)]
                    aug[(i,j)] = (fourier_augment_batch(src_imgs, sty, LAM), src_lbl)
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=DEVICE)
            orig_feats = []
            for i, (src_imgs, src_lbl) in enumerate(batches):
                feat  = model.extract(src_imgs, i)
                l_sup = arc_heads[i](feat, src_lbl)
                orig_feats.append(feat)
                loss += l_sup; ll["sup"] += l_sup.item()
                with torch.no_grad():
                    ep_corr  += arc_heads[i].get_logits(feat).argmax(1).eq(src_lbl).sum().item()
                    ep_total += src_lbl.size(0)
                for j in range(N):
                    if i == j: continue
                    aug_imgs, aug_lbl = aug[(i,j)]
                    l_a = arc_heads[i](model.extract(aug_imgs, i), aug_lbl)
                    loss += l_a; ll["sup"] += l_a.item()
            for i in range(N):
                aug_hf, aug_ll = [], []
                for j in range(N):
                    if i == j: continue
                    aug_imgs, aug_lbl = aug[(i,j)]
                    aug_hf.append(model.extract_all(aug_imgs)); aug_ll.append(aug_lbl)
                l_con = ALPHA * consistent_loss(orig_feats[i], aug_hf)
                loss += l_con; ll["con"] += l_con.item()
                aug_avg = torch.stack(
                    [torch.stack(hf, 0).mean(0) for hf in aug_hf], 0).mean(0)
                pos, neg = sample_triplet_pairs(aug_avg, aug_ll[0], batches[i][1])
                l_dt  = BETA * triplet_loss_fn(orig_feats[i], pos, neg, TRIPLET_MARGIN)
                loss += l_dt; ll["dt"] += l_dt.item()
            for i in range(N):
                for j in range(i+1, N):
                    l_ada = mkmmd_loss(orig_feats[i], orig_feats[j])
                    loss += l_ada; ll["ada"] += l_ada.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()
            ll["total"] += loss.item()
        for k in ll: ll[k] /= steps_per_epoch
        avg_acc = 100.0 * ep_corr / ep_total if ep_total > 0 else 0.0
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            cur_eer, cur_rank1 = evaluate(
                model, gal_loader, prb_loader, rst_eval, f"P2_ep{epoch+1:04d}")
            marker = "  *** new best ***" if cur_rank1 > best_rank1 else ""
            log(f"  P2 ep {epoch+1:03d}/{EPOCHS}  loss={ll['total']:.4f}  acc={avg_acc:.2f}%  "
                f"sup={ll['sup']:.4f} ada={ll['ada']:.4f} con={ll['con']:.4f} dt={ll['dt']:.4f}  "
                f"EER={cur_eer*100:.4f}%  Rank-1={cur_rank1:.2f}%{marker}")
            if cur_rank1 > best_rank1:
                best_rank1 = cur_rank1
                torch.save({"epoch": PRETRAIN_EPOCHS+epoch+1, "model": model.state_dict(),
                            "arc_heads": [h.state_dict() for h in arc_heads],
                            "rank1": cur_rank1, "eer": cur_eer}, ckpt_path)
        elif (epoch + 1) % (EVAL_EVERY // 2) == 0:
            log(f"  P2 ep {epoch+1:03d}/{EPOCHS}  loss={ll['total']:.4f}  acc={avg_acc:.2f}%")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_eer, final_rank1 = evaluate(model, gal_loader, prb_loader, rst_eval, "FINAL")
    log(f"  Best Rank-1={best_rank1:.2f}%  Final: EER={final_eer*100:.4f}%  Rank-1={final_rank1:.2f}%")
    return final_eer, final_rank1


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_summary(all_results, out_path):
    col_w  = 14
    header = (f"{'Setting':<22}{'Train domain':<38}{'Test domain':<26}"
              f"{'EER (%)':>{col_w}}{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)
    lines = ["\nCross-Domain Closed-Set Results — Palm-Auth (PDFG)", sep, header, sep]
    for r in all_results:
        eer_str   = f"{r['eer']*100:.2f}" if r['eer']   is not None else "—"
        rank1_str = f"{r['rank1']:.2f}"   if r['rank1'] is not None else "—"
        lines.append(f"{r['setting']:<22}{r['train_desc']:<38}{r['test_desc']:<26}"
                     f"{eer_str:>{col_w}}{rank1_str:>{col_w}}")
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f: f.write(text + "\n")
    log(f"Summary saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    log("=" * 72)
    log(f"PDFG — Cross-Domain Closed-Set (Palm-Auth)")
    log(f"Device  : {DEVICE}")
    log(f"N heads : {N}  (condition-based D1/D2 split)")
    log(f"D1 conds: {D1_CONDITIONS}")
    log(f"D2 conds: {D2_CONDITIONS}")
    log(f"Epochs  : Pretrain={PRETRAIN_EPOCHS}  Main={EPOCHS}")
    log(f"Settings: 2 scanner + {len(PAIRED_CONDITIONS)} paired-condition")
    log(f"Results : {SAVE_DIR}")
    log("=" * 72)

    log("\nScanning dataset …")
    cond_paths    = _collect_perspective(PALM_AUTH_ROOT)
    scanner_paths = _collect_scanner(PALM_AUTH_ROOT, SCANNER_SPECTRA)
    log(f"Perspective conditions: {sorted(cond_paths.keys())}")
    log(f"Scanner identities   : {len(scanner_paths)}")

    all_splits = load_or_generate_closedset_splits(
        cond_paths, scanner_paths, TEST_GALLERY_RATIO, SEED)

    SETTINGS = []
    SETTINGS.append({"tag": "setting_scanner", "label": "S_scanner",
                     "train_desc": "Perspective (all 190 IDs)",
                     "test_desc": "Scanner 50/50 gallery/probe",
                     "parser": lambda: parse_setting_scanner(
                         cond_paths, scanner_paths, all_splits, SEED)})
    SETTINGS.append({"tag": "setting_scanner_to_persp", "label": "S_scanner_to_persp",
                     "train_desc": "Scanner (148 IDs)",
                     "test_desc": "Perspective 50/50 gallery/probe",
                     "parser": lambda: parse_setting_scanner_to_perspective(
                         cond_paths, scanner_paths, all_splits, SEED)})

    conditions_found = sorted(cond_paths.keys())
    for cond_a, cond_b in PAIRED_CONDITIONS:
        if cond_a not in conditions_found or cond_b not in conditions_found:
            log(f"  [WARN] '{cond_a}' or '{cond_b}' not found — skipping"); continue
        ca, cb = cond_a, cond_b
        SETTINGS.append({"tag": f"setting_{ca}_{cb}", "label": f"S_{ca}_{cb}",
                         "train_desc": f"Perspective(not {ca}/{cb}) + Scanner",
                         "test_desc": f"gallery:{ca} / probe:{cb}",
                         "parser": (lambda ca=ca, cb=cb: parse_setting_paired_conditions(
                             ca, cb, cond_paths, scanner_paths, SEED))})

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
            eer, rank1 = run_experiment(d1_s, d2_s, gal_s, prb_s, n_cls, results_dir)
            elapsed    = time.time() - t_start
            log(f"\n  ✓  {s['label']}:  EER={eer*100:.4f}%  "
                f"Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            with open(os.path.join(results_dir, "results.json"), "w") as f:
                json.dump({"setting": s["label"], "train_desc": s["train_desc"],
                           "test_desc": s["test_desc"], "num_classes": n_cls,
                           "EER_pct": eer*100, "Rank1_pct": rank1}, f, indent=2)
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"], "eer": eer, "rank1": rank1})
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"\n  ✗  {s['label']} FAILED: {e}")
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"], "eer": None, "rank1": None})

    log(f"\n\n{'='*72}")
    log(f"ALL {len(SETTINGS)} SETTINGS COMPLETE")
    log(f"{'='*72}")
    print_and_save_summary(all_results, os.path.join(SAVE_DIR, "results_summary.txt"))


if __name__ == "__main__":
    main()
