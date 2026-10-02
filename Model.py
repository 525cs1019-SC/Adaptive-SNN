"""
FULL PIPELINE: Spiking ResNet on CIFAR-10 + Adaptive Timestep Gate
=====================================================================
Cell 1: Config, Data, Model definitions
Cell 2: Resumable training (TET loss)
Cell 3: Collect logits (multi-seed, for evaluation)
Cell 4: Adaptive exit comparison - Fixed-T vs SEENN-style vs stable-k vs learned gate
"""
import os, glob, shutil, time, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision, torchvision.transforms as T

# ---------------- Config ----------------
class Cfg:
    CIFAR_ROOT = "/kaggle/input/datasets/passiveacoustics/cifar-10-testsc"
    OUT_DIR    = "/kaggle/working"
    VAL_SPLIT  = 0.1
    CHANNELS   = [64, 128, 256]
    BLOCKS     = [2, 2, 2]
    DROPOUT    = 0.1
    MAX_T      = 6
    LEAK       = 0.9
    ENCODING   = "poisson"      # "poisson" or "direct"
    LOSS_MODE  = "tet"          # "tet" (all timesteps) or "final" (ablation)
    BATCH_SIZE = 192
    EPOCHS     = 100
    LR, WD, MIN_LR = 1e-3, 5e-4, 1e-5
    AMP        = True
    NUM_WORKERS = 4
    EVAL_EVERY = 2
    PATIENCE   = 12
    SEED       = 42
    N_CLASSES  = 10
cfg = Cfg()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(cfg.SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f"Device: {device} | GPUs: {n_gpu}")
for i in range(n_gpu):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

assert os.path.exists(cfg.CIFAR_ROOT), f"Dataset not found at {cfg.CIFAR_ROOT}"

# ---------------- Data ----------------
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
clean_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

ds_aug   = torchvision.datasets.CIFAR10(cfg.CIFAR_ROOT, train=True,  download=False, transform=train_tf)
ds_clean = torchvision.datasets.CIFAR10(cfg.CIFAR_ROOT, train=True,  download=False, transform=clean_tf)
test_set = torchvision.datasets.CIFAR10(cfg.CIFAR_ROOT, train=False, download=False, transform=clean_tf)

perm = torch.randperm(len(ds_aug), generator=torch.Generator().manual_seed(cfg.SEED)).tolist()
n_val = int(len(perm) * cfg.VAL_SPLIT)
val_idx, train_idx = perm[:n_val], perm[n_val:]
train_set = Subset(ds_aug, train_idx)      # augmented
val_set   = Subset(ds_clean, val_idx)      # clean transform (no train-time augmentation leakage)

kw = dict(num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=True)
train_loader = DataLoader(train_set, cfg.BATCH_SIZE, shuffle=True, drop_last=True, **kw)
val_loader   = DataLoader(val_set, 500, shuffle=False, **kw)
test_loader  = DataLoader(test_set, 500, shuffle=False, **kw)
print(f"Train {len(train_set)} | Val {len(val_set)} | Test {len(test_set)}")

# ---------------- Surrogate spike + LIF ----------------
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, alpha):
        ctx.save_for_backward(v); ctx.alpha = alpha
        return (v >= 1.0).to(v.dtype)
    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (1.0 + ctx.alpha * (v - 1.0).abs()) ** 2, None

class LIF(nn.Module):
    def __init__(self, leak, alpha=2.0):
        super().__init__()
        self.leak, self.alpha, self.v = leak, alpha, None
    def reset(self): self.v = None
    def forward(self, x):
        dt = x.dtype; x = x.float()
        v = x if self.v is None else self.leak * self.v + x
        s = SpikeFn.apply(v, self.alpha)
        self.v = v * (1.0 - s)
        return s.to(dt)

# ---------------- Spiking ResNet ----------------
class Block(nn.Module):
    def __init__(self, cin, cout, stride, p, leak):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.bn2 = nn.BatchNorm2d(cout)
        self.lif1, self.lif2 = LIF(leak), LIF(leak)
        self.drop = nn.Dropout2d(p) if p > 0 else nn.Identity()
        self.shortcut = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        out = self.lif1(self.drop(self.bn1(self.conv1(x))))
        out = self.drop(self.bn2(self.conv2(out)))
        return self.lif2(out + self.shortcut(x))

class SpikingResNet(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        self.T, self.enc, self.amp = cfg.MAX_T, cfg.ENCODING, cfg.AMP
        C = cfg.CHANNELS
        self.stem = nn.Sequential(nn.Conv2d(3, C[0], 3, padding=1, bias=False), nn.BatchNorm2d(C[0]))
        self.stem_lif = LIF(cfg.LEAK)
        blocks, cin = [], C[0]
        for si, cout in enumerate(C):
            for bi in range(cfg.BLOCKS[si]):
                stride = 2 if (si > 0 and bi == 0) else 1
                blocks.append(Block(cin, cout, stride, cfg.DROPOUT, cfg.LEAK)); cin = cout
        self.stages = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, n_classes)   # built in __init__ so optimizer trains it

    def reset(self):
        for m in self.modules():
            if isinstance(m, LIF): m.reset()

    def encode(self, x):
        if self.enc == "poisson":
            return (torch.rand_like(x) < torch.sigmoid(x)).float()
        return x

    def forward(self, x):
        """Returns running-average logits at every timestep: [B, T, n_classes]"""
        self.reset()
        run, outs = 0, []
        with torch.autocast("cuda", dtype=torch.float16, enabled=(self.amp and x.is_cuda)):
            for t in range(self.T):
                h = self.stem_lif(self.stem(self.encode(x)))
                h = self.stages(h)
                logits = self.fc(F.adaptive_avg_pool2d(h, 1).flatten(1))
                run = run + logits.float()
                outs.append(run / (t + 1))
        self.reset()
        return torch.stack(outs, 1)

def build_model():
    m = SpikingResNet(cfg, cfg.N_CLASSES).to(device)
    if n_gpu > 1: m = nn.DataParallel(m)
    return m

def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m

CKPT = f"{cfg.OUT_DIR}/best_{cfg.LOSS_MODE}_{cfg.ENCODING}.pth"
LAST = f"{cfg.OUT_DIR}/last_{cfg.LOSS_MODE}_{cfg.ENCODING}.pth"
print("Setup complete. Checkpoint target:", CKPT)
