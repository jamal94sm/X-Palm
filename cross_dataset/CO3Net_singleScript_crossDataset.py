"""
CO3Net — Cross-Dataset Palmprint Recognition
==================================================
Architecture and training method: unchanged from official CO3Net.
  - Dual LGC + CoordAtt + soft-argmax + PPU (CompetitiveBlock)
  - FC: 17328 → 4096 → 2048 + ArcFace
  - Loss: ce_weight * CrossEntropy + con_weight * SupConLoss
  - Paired-image training

Datasets
--------
  CASIA-MS  : {subjectID}_{spectrum}_{handSide}_{iter}.jpg  (flat folder)
  Palm-Auth : {ID}/roi_perspective/ + optional roi_scanner/
  MPDv2     : {subject}_{session}_{device}_{hand}_{iter}.jpg (flat folder)
  XJTU      : {device}/{condition}/{hand}_{id}/*.jpg
              devices    : huawei | iPhone
              conditions : Flash  | Nature
              IDs        : L_001…L_100  R_001…R_100  (200 total, select 190)

Two-group sampling (mirrors Palm-Auth natural distribution)
-----------------------------------------------------------
  CASIA-MS / MPDv2 : 150 IDs × 30  +  40 IDs × 15  = 190 IDs
  XJTU             : 150 IDs × 29  +  40 IDs × 15  = 190 IDs
                     (29 = 4 variations × ~7 images; 15 = 4 × ~4)
                     images drawn near-uniformly across the 4 variations
                     Flash-iPhone | Nature-iPhone | Flash-Huawei | Nature-Huawei

Combined evaluation set  (combined_evaluation_set = True)
---------------------------------------------------------
  Each of the FOUR datasets contributes 20% of its 190 selected IDs.
  All held-out IDs are merged with new global sequential labels.
  Two caches guarantee identical conditions across all experiments:
    combined_eval_cache.json          ← gallery / probe / train_remaining
    init_weights_{model}_nc{N}.pth   ← fixed initial weights
"""

# ==============================================================
#  CONFIG  — edit only this block
# ==============================================================
CONFIG = {
    # ── Dataset selection ──────────────────────────────────────
    # Choices: "CASIA-MS" | "Palm-Auth" | "MPDv2" | "XJTU"
    "train_data"           : "Palm-Auth",
    "test_data"            : "MPDv2",   # used only when combined_evaluation_set=False

    # ── Dataset paths ──────────────────────────────────────────
    "casiams_data_root"    : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "palm_auth_data_root"  : "/home/pai-ng/Jamal/smartphone_data",
    "mpd_data_root"        : "/home/pai-ng/Jamal/MPDv2_mediapipe_manual_roi",
    "xjtu_data_root"       : "/home/pai-ng/Jamal/XJTU-UP",

    # ── Splitting ──────────────────────────────────────────────
    "train_subject_ratio"  : 0.80,
    "test_gallery_ratio"   : 0.50,

    # ── Palm-Auth toggle ───────────────────────────────────────
    "use_scanner"          : True,

    # ── Combined evaluation set ────────────────────────────────
    "combined_evaluation_set"  : True,
    "combined_gallery_ratio"   : 0.50,
    "combined_eval_cache_path" : "./combined_eval_cache.json",

    # ── Model ──────────────────────────────────────────────────
    "img_side"             : 128,
    "dropout"              : 0.5,
    "arcface_s"            : 20.0,
    "arcface_m"            : 0.30,

    # ── Loss ───────────────────────────────────────────────────
    "ce_weight"            : 0.8,
    "con_weight"           : 0.2,
    "temperature"          : 0.07,

    # ── Training ───────────────────────────────────────────────
    "batch_size"           : 256,
    "num_epochs"           : 100,
    "lr"                   : 0.001,
    "lr_step"              : 30,
    "lr_gamma"             : 0.6,
    "augment_factor"       : 4,

    # ── Misc ───────────────────────────────────────────────────
    "results_dir"          : "./rst_co3net",
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

ALLOWED_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

# Two-group sampling constants (CASIA-MS / MPDv2 / Palm-Auth)
N_HIGH      = 150
N_LOW       = 40
TARGET_HIGH_CASIA = 29
TARGET_LOW_CASIA  = 15
TARGET_HIGH_MPD = 33
TARGET_LOW_MPD  = 16

# XJTU-specific targets (4 variations × ~7 = 28–29 for high group)
XJTU_N_TOTAL      = 200   # available IDs in the dataset
XJTU_N_SELECT     = 190   # how many to use
XJTU_TARGET_HIGH  = 30    # images per ID in high group
XJTU_TARGET_LOW   = 14    # images per ID in low group
# 4 variations: Flash-iPhone | Nature-iPhone | Flash-Huawei | Nature-Huawei
XJTU_VARIATIONS   = [
    ("iPhone",  "Flash"),
    ("iPhone",  "Nature"),
    ("huawei",  "Flash"),
    ("huawei",  "Nature"),
]


# ══════════════════════════════════════════════════════════════
#  SUPCONLOSS  (exact copy — unchanged)
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature      = temperature
        self.contrast_mode    = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]; anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature; anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits  = torch.exp(logits) * logits_mask
        log_prob    = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


# ══════════════════════════════════════════════════════════════
#  CO3NET ARCHITECTURE  (exact copy — unchanged)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = init_ratio if init_ratio > 0 else 1.0
        self.kernel      = 0
        self._SIGMA = 9.2  * self.init_ratio
        self._FREQ  = 0.057 / self.init_ratio
        self._GAMMA = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]))
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]))
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([self._FREQ]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out,
                     sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2; ymax = kernel_size // 2
        xmin = -xmax;            ymin = -ymax
        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)
        x = x.float().to(sigma.device); y = y.float().to(sigma.device)
        x_theta =  x * torch.cos(theta.view(-1,1,1,1)) + y * torch.sin(theta.view(-1,1,1,1))
        y_theta = -x * torch.sin(theta.view(-1,1,1,1)) + y * torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta)**2 + y_theta**2) / (8 * sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2 * math.pi * f.view(-1,1,1,1) * x_theta + psi.view(-1,1,1,1))
        gb = gb - gb.mean(dim=[2,3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in,
                                   self.channel_out, self.sigma, self.gamma,
                                   self.theta, self.f, self.psi)
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x): return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x): return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)
        y   = torch.cat([x_h, x_w], dim=2)
        y   = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0,1,3,2)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class CompetitiveBlock(nn.Module):
    """CO3Net Competitive Block: dual LGC + CoordAtt + soft-argmax + PPU."""
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock, self).__init__()
        self.gabor_conv2d  = GaborConv2d(channel_in, n_competitor, ksize,
                                         stride, padding, init_ratio)
        self.gabor_conv2d2 = GaborConv2d(n_competitor, n_competitor, ksize,
                                         1, ksize // 2, init_ratio)
        self.cooratt1 = CoordAtt(n_competitor, n_competitor)
        self.cooratt2 = CoordAtt(n_competitor, n_competitor)
        self.a       = nn.Parameter(torch.FloatTensor([1]))
        self.b       = nn.Parameter(torch.FloatTensor([0]))
        self.argmax  = nn.Softmax(dim=1)
        self.conv1   = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.cooratt1(self.gabor_conv2d(x))
        x = self.cooratt2(self.gabor_conv2d2(x))
        x = self.argmax((x - self.b) * self.a)
        return self.conv2(self.maxpool(self.conv1(x)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s; self.m = m
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine) if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))
        return self.s * F.linear(F.normalize(input), F.normalize(self.weight))


class co3net(nn.Module):
    """
    CO3Net = CB1 // CB2 // CB3 + FC(17328→4096→2048) + Dropout + ArcFace.
    Contrastive feature: L2-normalised 6144-d (4096+2048 concat).
    Matching embedding : L2-normalised 2048-d.
    """
    def __init__(self, num_classes, dropout=0.5,
                 arcface_s=20.0, arcface_m=0.30):
        super(co3net, self).__init__()
        self.num_classes = num_classes
        self.cb1 = CompetitiveBlock(1, 9,  35, 3, 17, init_ratio=1,    o2=12)
        self.cb2 = CompetitiveBlock(1, 36, 17, 3, 8,  init_ratio=0.5,  o2=24)
        self.cb3 = CompetitiveBlock(1, 9,  7,  3, 3,  init_ratio=0.25, o2=12)
        self.fc   = nn.Linear(17328, 4096)
        self.fc1  = nn.Linear(4096, 2048)
        self.drop = nn.Dropout(p=dropout)
        self.arclayer = ArcMarginProduct(2048, num_classes,
                                         s=arcface_s, m=arcface_m)

    def forward(self, x, y=None):
        x1 = self.cb1(x).view(x.shape[0], -1)
        x2 = self.cb2(x).view(x.shape[0], -1)
        x3 = self.cb3(x).view(x.shape[0], -1)
        x  = torch.cat((x1, x2, x3), dim=1)
        x1 = self.fc(x)
        x  = self.fc1(x1)
        fe = F.normalize(torch.cat((x1, x), dim=1), dim=-1)
        x  = self.arclayer(self.drop(x), y)
        return x, fe

    @torch.no_grad()
    def get_embedding(self, x):
        x1 = self.cb1(x).view(x.shape[0], -1)
        x2 = self.cb2(x).view(x.shape[0], -1)
        x3 = self.cb3(x).view(x.shape[0], -1)
        x  = torch.cat((x1, x2, x3), dim=1)
        x  = self.fc1(self.fc(x))
        return F.normalize(x, p=2, dim=1)


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
    """150 IDs × TARGET_HIGH  +  40 IDs × TARGET_LOW, evenly across spectra."""
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
    print(f"    High ({N_HIGH}×~{TARGET_HIGH}): min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f}")
    print(f"    Low  ({N_LOW}×~{TARGET_LOW}):  min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f}")
    return id2paths


def parse_palm_auth_data(data_root, use_scanner=False):
    """roi_perspective + optional roi_scanner (ALLOWED_SPECTRA only)."""
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
                    identity = subject_id + "_" + parts[1].lower()
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
    """Top-150 IDs × TARGET_HIGH  +  next-40 IDs (≥TARGET_LOW) × TARGET_LOW."""
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
                     len(id_dev[i].get("m",[]))) >= TARGET_LOW]
    if len(low_cands) < N_LOW:
        raise ValueError(f"Not enough IDs with ≥{TARGET_LOW} samples: "
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
    print(f"    High ({N_HIGH}×~{TARGET_HIGH}): "
          f"min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f} cutoff={cutoff_h}")
    print(f"    Low  ({N_LOW}×~{TARGET_LOW}):  "
          f"min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f} cutoff={cutoff_l}")
    return id2paths


def parse_xjtu_data(data_root, seed=42):
    """
    XJTU-UP: 200 IDs (L_001…L_100, R_001…R_100), 4 variations each.
    Structure: {data_root}/{device}/{condition}/{hand}_{id}/*.jpg

    Select XJTU_N_SELECT (190) IDs from 200, then:
      High group : 150 IDs × XJTU_TARGET_HIGH (29) images
                   sampled near-uniformly from 4 variations
      Low  group :  40 IDs × XJTU_TARGET_LOW  (15) images
                   sampled near-uniformly from 4 variations

    Identity key: folder name e.g. "L_001", "R_100"
    """
    rng     = random.Random(seed)
    IMG_EXTS = {".jpg",".jpeg",".bmp",".png"}

    # id_var[identity][(device, condition)] = [path, ...]
    id_var  = defaultdict(lambda: defaultdict(list))

    for device, condition in XJTU_VARIATIONS:
        var_dir = os.path.join(data_root, device, condition)
        if not os.path.isdir(var_dir):
            print(f"  [XJTU] WARNING: variation folder not found: {var_dir}")
            continue
        for id_folder in sorted(os.listdir(var_dir)):
            id_dir = os.path.join(var_dir, id_folder)
            if not os.path.isdir(id_dir):
                continue
            # id_folder is like "L_001" or "R_100"
            parts = id_folder.split("_")
            if len(parts) < 2 or parts[0].upper() not in ("L", "R"):
                continue
            identity = id_folder   # keep original: "L_001", "R_100"
            for fname in sorted(os.listdir(id_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                id_var[identity][(device, condition)].append(
                    os.path.join(id_dir, fname))

    all_ids = sorted(id_var.keys())
    print(f"  [XJTU] Total IDs found: {len(all_ids)}")
    if len(all_ids) < XJTU_N_SELECT:
        raise ValueError(f"Need {XJTU_N_SELECT} IDs but only {len(all_ids)} found "
                         f"in {data_root}")

    # Randomly select XJTU_N_SELECT IDs from all available
    selected = sorted(rng.sample(all_ids, XJTU_N_SELECT))
    rng.shuffle(selected)
    high_ids = selected[:N_HIGH]   # 150
    low_ids  = selected[N_HIGH:]   # 40

    def _sample_from_variations(ident, target):
        """
        Sample `target` images near-uniformly across the 4 variations.
        Same logic as CASIA-MS spectra: base = target // 4, remainder
        distributed to the first `rem` variations (shuffled).
        """
        var_keys  = list(XJTU_VARIATIONS)
        rng.shuffle(var_keys)
        n_var     = len(var_keys)
        base_v    = target // n_var
        rem_v     = target %  n_var
        chosen    = []
        for j, vk in enumerate(var_keys):
            k         = base_v + (1 if j < rem_v else 0)
            available = id_var[ident].get(vk, [])
            k         = min(k, len(available))
            if k > 0:
                chosen.extend(rng.sample(available, k))
        return chosen

    id2paths = {}
    for ident in high_ids:
        id2paths[ident] = _sample_from_variations(ident, XJTU_TARGET_HIGH)
    for ident in low_ids:
        id2paths[ident] = _sample_from_variations(ident, XJTU_TARGET_LOW)

    actual = sum(len(v) for v in id2paths.values())
    hc = [len(id2paths[i]) for i in high_ids]
    lc = [len(id2paths[i]) for i in low_ids]

    print(f"  [XJTU] ids={len(id2paths)}  total={actual}")
    print(f"    High ({N_HIGH}×~{XJTU_TARGET_HIGH}): "
          f"min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f}")
    print(f"    Low  ({N_LOW}×~{XJTU_TARGET_LOW}):  "
          f"min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f}")
    print(f"    Variations : {[f'{d}/{c}' for d,c in XJTU_VARIATIONS]}")
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
    """
    Build or reload the combined evaluation set from ALL FOUR datasets.

    FIRST RUN  → parses all four datasets, holds out 20% of each
                 dataset's selected IDs, writes JSON cache.
    LATER RUNS → loads JSON cache; no parsing, no RNG consumed.

    Held-out IDs per dataset:
      CASIA-MS  : 20% of 190 = 38 IDs
      Palm-Auth : 20% of all IDs (natural distribution)
      MPDv2     : 20% of 190 = 38 IDs
      XJTU      : 20% of 190 = 38 IDs
    """
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

    # Parse all FOUR datasets
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

        # Fully independent per-dataset RNGs — isolated from global state
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
#  FIXED MODEL INITIALISATION  (.pth cache)
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(net, cfg, num_classes, device):
    """Save init weights on first run, load on every subsequent run.
    Filename includes model name to avoid cross-model conflicts."""
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
#  SPLITS  (standard mode)
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

class PairedDataset(Dataset):
    """Paired-image training dataset for CO3Net (augment_factor × expansion)."""
    def __init__(self, samples, img_side=128, augment_factor=1):
        self.samples        = samples
        self.augment_factor = augment_factor
        self.label2idxs     = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)

        self.aug_transform = T.Compose([
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
        real_idx     = index % len(self.samples)
        path1, label = self.samples[real_idx]
        idxs = self.label2idxs[label]
        idx2 = real_idx
        while idx2 == real_idx and len(idxs) > 1:
            idx2 = random.choice(idxs)
        path2 = self.samples[idx2][0]
        img1 = self.aug_transform(Image.open(path1).convert("L"))
        img2 = self.aug_transform(Image.open(path2).convert("L"))
        return [img1, img2], label


class SingleDataset(Dataset):
    """Deterministic single-image dataset for gallery and probe."""
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([
            T.Resize(img_side), T.ToTensor(), NormSingleROI(outchannels=1)])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, con_criterion,
                  optimizer, device, phase,
                  ce_weight=0.8, con_weight=0.2):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for datas, target in loader:
            data1  = datas[0].to(device)
            data2  = datas[1].to(device)
            target = target.to(device)

            if is_train: optimizer.zero_grad()

            output,  fe1 = model(data1, target if is_train else None)
            output2, fe2 = model(data2, target if is_train else None)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

            ce_loss  = criterion(output, target)
            con_loss = con_criterion(fe, target)
            loss     = ce_weight * ce_loss + con_weight * con_loss

            if is_train: loss.backward(); optimizer.step()

            running_loss    += loss.item() * data1.size(0)
            running_correct += output.data.max(1)[1].eq(target).sum().item()
            total           += data1.size(0)

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


def compute_eer(scores_array):
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return 1.0, 0.0
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, thresh


def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    probe_feats,   probe_labels   = extract_features(model, probe_loader, device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe = len(probe_feats)

    sim_matrix = probe_feats @ gallery_feats.T

    scores_list, labels_list = [], []
    for i in range(n_probe):
        for j in range(sim_matrix.shape[1]):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer(scores_arr)

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]]
                  for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")
    _save_roc_det(scores_arr, out_dir, tag)

    print(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


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
    ce_weight           = CONFIG["ce_weight"]
    con_weight          = CONFIG["con_weight"]
    temperature         = CONFIG["temperature"]
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
    print(f"  CO3Net Palmprint Recognition")
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
    print(f"  Sampling       : high={N_HIGH}×{TARGET_HIGH}  low={N_LOW}×{TARGET_LOW}  "
          f"(XJTU high×{XJTU_TARGET_HIGH})")
    print(f"  Loss           : {ce_weight}×CE + {con_weight}×SupCon(τ={temperature})")
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
        PairedDataset(train_samples, img_side, augment_factor),
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
    print(f"Building CO3Net — num_classes={num_classes} …")
    net = co3net(num_classes=num_classes, dropout=dropout,
                 arcface_s=arcface_s, arcface_m=arcface_m)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        net = DataParallel(net)

    if use_combined_eval:
        net = get_or_create_init_weights(net, CONFIG, num_classes, device)
    else:
        print("  Training from scratch (random init).")

    if CONFIG.get("resume", False):
        for ckpt in ["net_params_best_eer.pth", "net_params_best.pth",
                     "net_params.pth"]:
            path = os.path.join(results_dir, ckpt)
            if os.path.exists(path):
                _net = net.module if isinstance(net, DataParallel) else net
                _net.load_state_dict(torch.load(path, map_location=device))
                print(f"  Resumed from : {path}"); break
        else:
            print("  No checkpoint found — starting from init weights.")

    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=temperature,
                               base_temperature=temperature)
    optimizer     = optim.Adam(net.parameters(), lr=lr)
    scheduler     = lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

    # ── training loop ─────────────────────────────────────────────────────
    train_losses, train_accs = [], []
    best_eer = 1.0; last_eer = float("nan"); last_rank1 = float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 evaluated every {eval_every} epochs.\n")

    if CONFIG.get("eval_only", False):
        print("  eval_only=True — skipping training.\n")
    else:
        # ── pre-training evaluation ────────────────────────────────────────
        _net = net.module if isinstance(net, DataParallel) else net
        print("  Pre-training evaluation (before any gradient update) …")
        cur_eer, cur_rank1 = evaluate(
            _net, probe_loader, gallery_loader,
            device, out_dir=rst_eval, tag=f"ep-001_pretrain_{eval_tag_base}")
        best_eer = cur_eer; last_eer = cur_eer; last_rank1 = cur_rank1
        torch.save(_net.state_dict(),
                   os.path.join(results_dir, "net_params_best_eer.pth"))
        print(f"  *** Initial best EER: {best_eer*100:.4f}% ***\n")

        for epoch in range(num_epochs):
            t_loss, t_acc = run_one_epoch(
                net, train_loader, criterion, con_criterion,
                optimizer, device, "training",
                ce_weight=ce_weight, con_weight=con_weight)
            scheduler.step()

            train_losses.append(t_loss); train_accs.append(t_acc)
            _net = net.module if isinstance(net, DataParallel) else net

            if (epoch % eval_every == 0 and epoch > 0) or epoch == num_epochs - 1:
                tag = f"ep{epoch:04d}_{eval_tag_base}"
                cur_eer, cur_rank1 = evaluate(
                    _net, probe_loader, gallery_loader,
                    device, out_dir=rst_eval, tag=tag)
                last_eer, last_rank1 = cur_eer, cur_rank1
                if cur_eer < best_eer:
                    best_eer = cur_eer
                    torch.save(_net.state_dict(),
                               os.path.join(results_dir, "net_params_best_eer.pth"))
                    print(f"  *** New best EER: {best_eer*100:.4f}% ***")

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                ts        = time.strftime("%H:%M:%S")
                eer_str   = f"{last_eer*100:.4f}%"  if not math.isnan(last_eer)   else "N/A"
                rank1_str = f"{last_rank1:.2f}%"     if not math.isnan(last_rank1) else "N/A"
                print(f"[{ts}] ep {epoch:04d} | "
                      f"loss={t_loss:.5f} | acc={t_acc:.2f}% | "
                      f"EER={eer_str}  Rank-1={rank1_str}")

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

    saved_name = (f"CO3Net_train{train_data.replace('-','').replace(' ','')}"
                  f"_eval{eval_tag_base}.pth")
    torch.save(eval_net.state_dict(), os.path.join(results_dir, saved_name))
    print(f"  Model saved as {saved_name}")

    final_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag=f"FINAL_{eval_tag_base}")

    print(f"\n{'='*60}")
    print(f"  Train  : {train_data} ({n_train_ids} subjects, {n_train_imgs} imgs)")
    print(f"  Eval   : "
          f"{'combined (CASIA-MS + Palm-Auth + MPDv2 + XJTU)' if use_combined_eval else test_data}")
    print(f"  FINAL EER    : {final_eer*100:.4f}%")
    print(f"  FINAL Rank-1 : {final_rank1:.3f}%")
    print(f"  Results      : {results_dir}")
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
        f.write(f"Sampling           : "
                f"high={N_HIGH}×{TARGET_HIGH} low={N_LOW}×{TARGET_LOW} "
                f"(XJTU high×{XJTU_TARGET_HIGH})\n")
        f.write(f"Loss               : "
                f"{ce_weight}×CE + {con_weight}×SupCon(τ={temperature})\n")
        f.write(f"Num classes        : {num_classes}\n")
        f.write(f"Eval set           : "
                f"{'combined' if use_combined_eval else test_data}\n")
        f.write(f"Eval subjects      : {n_test_ids}\n")
        f.write(f"Gallery samples    : {len(gallery_samples)}\n")
        f.write(f"Probe samples      : {len(probe_samples)}\n")
        f.write(f"Final EER          : {final_eer*100:.6f}%\n")
        f.write(f"Final Rank-1       : {final_rank1:.3f}%\n")


if __name__ == "__main__":
    main()
