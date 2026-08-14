#!/usr/bin/env python3
"""
PredU-OD Pilot — Probe-Signal Verification (self-contained, Colab Pro ready)
============================================================================
Implements the Week 1-2 go/no-go pilot's five measurements and prints a verdict
against the gates in configs/pilot.json:

  1. GLIGEN 512px generation + box fidelity mu_Q      (gate: mu_Q >= 0.55)
  2. YOLO11s on a 10% COCO regime: baseline mAP, failure profile, EL2N-style
     early-training difficulty, corr(F,D) sanity      (gate: warn if >= 0.9)
  3. Probe signal: stratified CONTRAST probes (proposal §10 Stage C: sample
     subsets stratified over the feature space) -> spread of U(S) vs 2-seed
     noise floor, SNR, preliminary Kendall-tau of best univariate feature,
     and the mechanism contrast dU = U(HIGH-FN) - U(LOW-FN)
                                                      (gates: SNR >= 1.0, tau >= 0.3)

CLI (final signal test, compute-constrained):
  !pip install -q ultralytics pycocotools diffusers transformers
  !python pilot_probe_signal.py --mode full --frac 0.02 --epochs 12
    --frac 0.02   : 2% of COCO train2017 as D_real (data-sparse => max headroom)
    --epochs 12   : equal schedule for baseline AND probes (fair dU = dmAP)
  The pre-registered full protocol stays at 10% COCO / 60-30 epochs; the pilot
  shrinks the regime so the go/no-go signal test fits ~1.5 GPU-h.
  4. DMR feasibility note (reported, not executed here)
  5. GPU-hours per probe train -> projected n_probes (gate: <= 0.6 h/probe)

Usage (Colab):
  !pip install -q ultralytics pycocotools          # torch is preinstalled
  !python pilot_probe_signal.py --mode full        # real GLIGEN + 10% COCO pilot
  !python pilot_probe_signal.py --mode smoke --quick   # <30 min pipeline smoke test

Design notes / honesty guards:
  * The real validation set is ONLY ever used for evaluation (never for
    generation, selection, feature fitting, or probe construction).
  * 'Difficulty' here is the PILOT proxy: early-training detection difficulty
    (miss rate + low matched confidence at epochs 2-5). The exact EL2N
    formulation is locked in the pre-registration for the full protocol; the
    proxy is labeled as such in every output.
  * D_CLIP is proxied by D_layout (box-layout diversity) in the pilot; CLIP
    embeddings are added in the full run. Labeled as such in output.
  * If GLIGEN cannot be loaded (no diffusers / no disk), the script falls back
    to a REAL-image stand-in pool so the probe/utility/statistics machinery is
    still exercised end to end (smoke). The scientific measurement requires
    the real GLIGEN pool (mode=full).
  * All intermediate results are cached to results/*.json; rerun with --force
    to recompute. A dead Colab session resumes from cache.

Dataset layout (YOLO convention, created by this script):
  data/coco10/images/train/*.jpg   data/coco10/labels/train/*.txt
  data/coco10/images/val/*.jpg     data/coco10/labels/val/*.txt
  data/pool/images/*.jpg           data/pool/labels/*.txt      (GLIGEN pool)
"""
import argparse, hashlib, json, math, os, random, shutil, sys, time
import urllib.request, zipfile
from pathlib import Path

SCRIPT_VERSION = "v10"  # v10: stratified CONTRAST probes (pre-registered design), --frac/--epochs, COCO ann-path fix, poolsrc reserve, corrupt-zip guard, full-mode preflight

# --------------------------------------------------------------------------
# 0. Config (merged with configs/pilot.json when present)
# --------------------------------------------------------------------------
DEFAULTS = {
    "paths": {
        "coco_ann": "data/instances_train2017.json",
        "subset_out": "data/coco10",
        "pool_out": "data/pool",
        "runs_out": "runs",
        "results_out": "results",
    },
    "coco_subset": {"fraction": 0.1, "seed": 42, "min_anns_per_image": 1},
    "generation": {"generator": "gligen", "resolution": 512,
                   "model_id": "masterful/gligen-1-4-generation-text-box",
                   "n_images": 300, "n_pools": 3, "seed": 7,
                   "strata": ["class", "scale", "count", "context"]},
    "detector": {"model_name": "yolo11s.pt", "imgsz": 512, "epochs_baseline": 60,
                 "epochs_probe": 30, "batch": 16, "seed": 42, "conf": 0.05,
                 "iou_thr": 0.5},
    "failure": {"match_iou": 0.5, "el2n_window_epoch": 5, "el2n_epochs_total": 30},
    "probes": {"sizes": [100, 150, 250], "n_subsets": 12, "n_seeds_duplicate": 2,
               "strata_col": "class_entropy"},
    "gates": {"box_fidelity_min": 0.55, "snr_min": 1.0, "corr_fd_warn": 0.9,
              "prelim_tau_min": 0.3, "gpu_hours_per_probe_max": 0.6,
              "probe_delta_noise_floor": 0.15},
}
def _merge(d, u):
    for k, v in u.items():
        d[k] = _merge(d.get(k, {}), v) if isinstance(v, dict) else v
    return d
CFG = _merge(json.loads(json.dumps(DEFAULTS)),
             json.loads(Path("configs/pilot.json").read_text())
             if Path("configs/pilot.json").exists() else {})
P, DET, FAIL, PROB, GATES = (CFG["paths"], CFG["detector"], CFG["failure"],
                             CFG["probes"], CFG["gates"])

COCO_NAMES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
 "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
 "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
 "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
 "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
 "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
 "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
 "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
 "clock","vase","scissors","teddy bear","hair drier","toothbrush"]
NAME2ID = {n: i for i, n in enumerate(COCO_NAMES)}

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

CURRENT_MODE = "smoke"  # set in main(); separates smoke/full caches

def config_fingerprint():
    """Hash of the config values that affect probe results. Stale caches from
    older script/config versions (different epochs, pool splits, subset sizes)
    are silently invalidated instead of being reused — a stale probe cache with
    a degenerate pool was producing repeated false NO-GO verdicts."""
    key = {"version": SCRIPT_VERSION, "mode": CURRENT_MODE,
           "epochs_probe": DET["epochs_probe"], "epochs_baseline": DET["epochs_baseline"],
           "batch": DET["batch"], "seed": DET["seed"], "imgsz": DET["imgsz"],
           "frac": CFG["coco_subset"]["fraction"], "n_subsets": PROB["n_subsets"],
           "sizes": PROB["sizes"]}
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]

# --------------------------------------------------------------------------
# 1. Pure-python stats (no scipy needed)
# --------------------------------------------------------------------------
def spearman(xs, ys):
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]: j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1): r[idx[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0

def kendall_tau(xs, ys):
    n = len(xs)
    if n < 2: return 0.0
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx * dy > 0: concord += 1
            elif dx * dy < 0: discord += 1
    return (concord - discord) / (n * (n - 1) / 2)

def entropy(counts):
    tot = sum(counts)
    if tot <= 0: return 0.0
    return -sum((c / tot) * math.log(c / tot) for c in counts if c > 0)

def vif(mat):
    """Variance inflation factor per column via numpy (preinstalled on Colab)."""
    import numpy as np
    X = np.asarray(mat, dtype=float)
    if X.shape[1] < 2: return [1.0] * X.shape[1]
    out = []
    for k in range(X.shape[1]):
        y, Xr = X[:, k], np.delete(X, k, axis=1)
        Xr = np.column_stack([np.ones(len(y)), Xr])
        try:
            beta, *_ = np.linalg.lstsq(Xr, y, rcond=None)
            resid = y - Xr @ beta
            r2 = 1.0 - (resid ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum())
        except Exception:
            r2 = 0.0
        out.append(1.0 / (1.0 - r2) if r2 < 0.9999 else float("inf"))
    return out

# --------------------------------------------------------------------------
# 2. Downloads (resumable, with progress)
# --------------------------------------------------------------------------
def download(url, dest, chunk=1 << 20):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"cache hit: {dest.name} ({dest.stat().st_size/1e6:.0f} MB)"); return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url} -> {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got, t0 = 0, time.time()
        while True:
            b = r.read(chunk)
            if not b: break
            f.write(b); got += len(b)
            if total and time.time() - t0 > 5:
                log(f"  {got/1e6:.0f}/{total/1e6:.0f} MB ({got/total*100:.0f}%)")
                t0 = time.time()
    log(f"done: {dest.name}")

def ensure_zip_ok(zp):
    """A partial/interrupted download silently poisons the run hours later (the
    19 GB train2017.zip crash would burn compute for nothing). Cheap check now:
    open the zip and read the central directory (testzip() would decompress the
    whole archive — far too slow for 19 GB). Per-file CRC is verified by
    extract_zip at extraction time anyway."""
    try:
        with zipfile.ZipFile(zp) as z:
            ok = len(z.namelist()) > 0
        if ok:
            return True
        log(f"empty zip {zp.name}; deleting, will re-download")
    except Exception as e:
        log(f"unreadable zip {zp.name} ({e}); deleting, will re-download")
    Path(zp).unlink(missing_ok=True)
    return False

def extract_zip(zp, members, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        todo = [m for m in members if not (out / Path(m).name).exists()]
        if not todo: return
        log(f"extracting {len(todo)} files from {Path(zp).name}")
        for i, m in enumerate(todo):
            with z.open(m) as src, open(out / Path(m).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if (i + 1) % 2000 == 0: log(f"  {i+1}/{len(todo)}")

# --------------------------------------------------------------------------
# 3. COCO setup (full: 10% train2017 + val2017 | smoke: coco128)
# --------------------------------------------------------------------------
def _write_yolo_labels(ids, imgs, anns, labdir):
    labdir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        im = imgs[i]; W, H = im["width"], im["height"]
        lines = []
        for a in anns.get(i, []):
            x, y, w, h = a["bbox"]
            cx, cy = x + w / 2, y + h / 2
            lines.append(f"{a['category_id']-1} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}")
        Path(labdir, im["file_name"].replace(".jpg", ".txt")).write_text("\n".join(lines))

def build_coco_full():
    """Downloads COCO once, builds a class-coverage-first train subset (fraction
    from config, default 10%) + full val2017, returns (train_img_paths,
    val_img_paths, root). A small 'poolsrc' split is reserved from train2017 so
    the GLIGEN-fallback stand-in pool is genuinely held out (never in D_real)."""
    root = Path(P["subset_out"]); root.mkdir(parents=True, exist_ok=True)
    ann_zip = Path("data") / "annotations_trainval2017.zip"
    train_zip, val_zip = Path("data") / "train2017.zip", Path("data") / "val2017.zip"
    download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip)
    download("http://images.cocodataset.org/zips/train2017.zip", train_zip)
    download("http://images.cocodataset.org/zips/val2017.zip", val_zip)
    for zp, url in ((ann_zip, "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
                    (train_zip, "http://images.cocodataset.org/zips/train2017.zip"),
                    (val_zip, "http://images.cocodataset.org/zips/val2017.zip")):
        if zp.exists() and not ensure_zip_ok(zp):
            download(url, zp)
    # annotations extract to data/annotations/*.json; coco_ann points at the
    # train json exactly where this code reads it (was data/instances_*.json —
    # a guaranteed FileNotFoundError after the 19 GB download; fixed)
    ann_path = Path(P["coco_ann"])
    val_ann_path = Path("data/annotations/instances_val2017.json")
    if not ann_path.exists():
        with zipfile.ZipFile(ann_zip) as z:
            z.extract("annotations/instances_train2017.json", "data")
            z.extract("annotations/instances_val2017.json", "data")
    if not ann_path.exists() and (Path("data/annotations/instances_train2017.json")).exists():
        Path("data/annotations/instances_train2017.json").replace(ann_path)
    ann = json.loads(ann_path.read_text())
    imgs = {im["id"]: im for im in ann["images"]}
    anns = {}
    for a in ann["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    rng = random.Random(CFG["coco_subset"]["seed"])
    valid = [i for i, a in anns.items() if len(a) >= CFG["coco_subset"]["min_anns_per_image"]]
    rng.shuffle(valid)
    need = max(1, int(len(valid) * CFG["coco_subset"]["fraction"]))
    chosen, seen = [], set()
    # pass 1: guarantee every class is represented (one image per class first),
    # otherwise the fill below saturates with the first category (person)
    for c in sorted({a["category_id"] for a in ann["annotations"]}):
        for i in valid:
            if i in seen: continue
            if any(a["category_id"] == c for a in anns[i]):
                chosen.append(i); seen.add(i)
                break
    # pass 2: random fill to reach the target
    for i in valid:
        if len(chosen) >= need: break
        if i not in seen:
            chosen.append(i); seen.add(i)
    chosen = chosen[:need]
    # reserve a small held-out pool-source split (GLIGEN-fallback stand-in only)
    n_poolsrc = min(150, max(1, int(len(chosen) * 0.05)))
    pool_ids = set(rng.sample(chosen, n_poolsrc))
    train_ids = [i for i in chosen if i not in pool_ids]
    log(f"{CFG['coco_subset']['fraction']:.0%} subset: {len(train_ids)} train + "
        f"{len(pool_ids)} pool-source images")
    tdir, vdir = root / "images/train", root / "images/val"
    tlab, vlab = root / "labels/train", root / "labels/val"
    pdir, plab = root / "images/poolsrc", root / "labels/poolsrc"
    # selective extraction of only the sampled images from the 19GB zip
    wanted = {f"train2017/{imgs[i]['file_name']}" for i in train_ids}
    wanted_pool = {f"train2017/{imgs[i]['file_name']}" for i in pool_ids}
    with zipfile.ZipFile(train_zip) as z:
        extract_zip(train_zip, [m for m in z.namelist() if m in wanted], tdir)
        extract_zip(train_zip, [m for m in z.namelist() if m in wanted_pool], pdir)
    with zipfile.ZipFile(val_zip) as z:
        extract_zip(val_zip, [m for m in z.namelist() if m.endswith(".jpg")], vdir)
    _write_yolo_labels(train_ids, imgs, anns, tlab)
    _write_yolo_labels(list(pool_ids), imgs, anns, plab)
    if not val_ann_path.exists():
        with zipfile.ZipFile(ann_zip) as z:
            z.extract("annotations/instances_val2017.json", "data")
    val_ann = json.loads(val_ann_path.read_text())
    vimgs = {im["id"]: im for im in val_ann["images"]}
    vanns = {}
    for a in val_ann["annotations"]:
        vanns.setdefault(a["image_id"], []).append(a)
    _write_yolo_labels([im["id"] for im in val_ann["images"]], vimgs, vanns, vlab)
    train_paths = sorted(str(p) for p in tdir.glob("*.jpg"))
    val_paths = sorted(str(p) for p in vdir.glob("*.jpg"))
    log(f"train images: {len(train_paths)}, val images: {len(val_paths)}")
    return train_paths, val_paths, root

def build_coco_smoke():
    """coco128 (128 imgs, ~6MB): 60 train / 40 pool-source / 28 val, same layout.
    The pool-source split is CRITICAL: the stand-in pool must contain images the
    detector has never trained on, otherwise every probe adds only duplicated
    data and U(S) spread is 0 by construction (false NO-GO)."""
    root = Path(P["subset_out"]); root.mkdir(parents=True, exist_ok=True)
    zp = Path("data") / "coco128.zip"
    download("https://ultralytics.com/assets/coco128.zip", zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall("data")
    src_img = Path("data/coco128/images/train2017")
    src_lab = Path("data/coco128/labels/train2017")
    imgs = sorted(src_img.glob("*.jpg"))
    rng = random.Random(42); rng.shuffle(imgs)
    splits = {"train": imgs[:60], "poolsrc": imgs[60:100], "val": imgs[100:]}
    for split, ims in splits.items():
        d_img = root / "images" / split
        d_lab = root / "labels" / split
        d_img.mkdir(parents=True, exist_ok=True); d_lab.mkdir(parents=True, exist_ok=True)
        for s in ims:
            shutil.copy(s, d_img / s.name)
            lf = src_lab / s.name.replace(".jpg", ".txt")
            if lf.exists(): shutil.copy(lf, d_lab / s.name.replace(".jpg", ".txt"))
    tdir, vdir = root / "images/train", root / "images/val"
    return (sorted(str(p) for p in tdir.glob("*.jpg")),
            sorted(str(p) for p in vdir.glob("*.jpg")), root)

def dataset_yaml(root, train, val, name):
    """Absolute train/val paths: Ultralytics joins relative values onto the yaml's
    'path' key, which would double the prefix (data/coco10/data/coco10/...)."""
    d = {"path": str(Path(root).resolve()),
         "train": str(Path(train).resolve()),
         "val": str(Path(val).resolve()),
         "names": {i: n for i, n in enumerate(COCO_NAMES)}}
    p = Path("data") / name
    p.write_text(json.dumps(d))
    return str(p)

# --------------------------------------------------------------------------
# 4. Labels, matching, failure profile
# --------------------------------------------------------------------------
def read_labels(txt):
    out = []
    if not Path(txt).exists(): return out
    for ln in Path(txt).read_text().splitlines():
        p = ln.split()
        if len(p) < 5: continue
        cls, cx, cy, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        out.append([cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out

def label_path_for(img_path):
    """Swap the 'images' path component for 'labels' (works for both
    data/<root>/images/<split>/x.jpg -> data/<root>/labels/<split>/x.txt
    and data/pool/images/x.jpg -> data/pool/labels/x.txt)."""
    p = Path(img_path)
    parts = list(p.parts)
    try:
        i = parts.index("images")
    except ValueError:
        return p.with_suffix(".txt")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")

def scale_gt_to_pixels(gt, img_path):
    from PIL import Image
    with Image.open(img_path) as im:
        W, H = im.size
    return [[g[0], g[1] * W, g[2] * H, g[3] * W, g[4] * H] for g in gt]

def iou(a, b):
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    ua = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0

def match(gt, pred, thr=0.5):
    matched, used_g, used_p = [], set(), set()
    for pi, p in enumerate(pred):
        best, bi = thr, -1
        for gi, g in enumerate(gt):
            if gi in used_g: continue
            v = iou(g, p)
            if v > best: best, bi = v, gi
        if bi >= 0:
            matched.append((bi, pi, best)); used_g.add(bi); used_p.add(pi)
    return (matched, [i for i in range(len(gt)) if i not in used_g],
            [i for i in range(len(pred)) if i not in used_p])

def image_failure(gt, pred, confs, thr=0.5):
    """Per-image failure stats (pre-training diagnostics only, never synthetic ΔAP)."""
    m, ug, up = match(gt, pred, thr)
    n_gt, n_p = max(1, len(gt)), max(1, len(pred))
    fn_rate = len(ug) / n_gt
    fp_rate = len(up) / n_p
    loc = [1 - v for _, _, v in m]
    loc_err = sum(loc) / len(loc) if loc else 0.0
    matched_conf = [confs[pi] for _, pi, _ in m]
    ce = entropy([round(c, 2) for c in matched_conf]) if matched_conf else 0.0
    areas = [(g[3] - g[1]) * (g[4] - g[2]) for g in gt]
    n_gt_a = max(1, len(gt))
    sm = sum(1 for a in areas if a < 0.005) / n_gt_a
    lg = sum(1 for a in areas if a > 0.04) / n_gt_a
    return {"fn_rate": fn_rate, "fp_rate": fp_rate, "loc_err": loc_err,
            "conf_entropy": ce, "frac_small": sm, "frac_large": lg,
            "n_gt": len(gt), "density": len(gt)}

# --------------------------------------------------------------------------
# 5. Early-training difficulty (EL2N-style proxy)
# --------------------------------------------------------------------------
def early_difficulty(baseline_runs_dir, train_paths, sample=100, epochs=(2, 3, 4, 5), seed=42):
    """Mean over early epochs of (miss rate + 1 - matched confidence). PILOT PROXY."""
    from ultralytics import YOLO
    rng = random.Random(seed)
    sample = rng.sample(train_paths, min(sample, len(train_paths)))
    per_img = {p: [] for p in sample}
    for ep in epochs:
        ck = Path(baseline_runs_dir) / f"weights/epoch{ep}.pt"
        if not ck.exists():
            log(f"  (no checkpoint epoch{ep}, skipping)"); continue
        m = YOLO(str(ck))
        res = m.predict(sample, imgsz=DET["imgsz"], conf=0.001, iou=0.5, verbose=False)
        for p, r in zip(sample, res):
            gt = scale_gt_to_pixels(read_labels(label_path_for(p)), p)
            boxes = r.boxes
            # detections are [x1,y1,x2,y2]; prepend dummy cls to match iou()'s
            # [cls,x1,y1,x2,y2] format used by GT boxes
            pred = [[0.0] + b.xyxy[0].tolist() for b in boxes] if boxes is not None else []
            confs = [float(b.conf) for b in boxes] if boxes is not None else []
            _, ug, _ = match(gt, pred, FAIL["match_iou"])
            miss = len(ug) / max(1, len(gt))
            mc = sum(confs) / len(confs) if confs else 0.0
            per_img[p].append(min(1.0, miss + (1.0 - mc)))
    return {p: (sum(v) / len(v)) if v else 1.0 for p, v in per_img.items()}

# --------------------------------------------------------------------------
# 6. Baseline train + eval
# --------------------------------------------------------------------------
def train_and_eval(data_yaml, name, epochs, seed, save_period=-1):
    from ultralytics import YOLO
    t0 = time.time()
    m = YOLO(DET["model_name"])
    m.train(data=data_yaml, epochs=epochs, imgsz=DET["imgsz"], batch=DET["batch"],
            device=0, project=P["runs_out"], name=name, seed=seed, save_period=save_period,
            plots=False, verbose=False, exist_ok=True)
    # Ultralytics nests runs under <project>/<task>/<name> and the task dir varies
    # by version; read the real save dir from the trainer instead of guessing.
    save_dir = Path(m.trainer.save_dir)
    best_path = str(save_dir / "weights/best.pt")
    best = YOLO(best_path)
    val = best.val(data=data_yaml, split="val", imgsz=DET["imgsz"], conf=0.001,
                   iou=0.5, verbose=False)
    return {"mAP": float(val.box.map), "mAP50": float(val.box.map50),
            "hours": (time.time() - t0) / 3600.0,
            "best": best_path, "save_dir": str(save_dir)}

# --------------------------------------------------------------------------
# 7. GLIGEN neutral pool (mode=full) / real-image stand-in pool (fallback)
# --------------------------------------------------------------------------
def iou4(a, b):
    """IoU for 4-elem [x1,y1,x2,y2] boxes (GLIGEN layout uses 4-elem; iou() uses
    the 5-elem [cls,x1,y1,x2,y2] format and would index out of range here)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def layout_sample(rng):
    """Neutral layout: random class/scale/count boxes (no failure conditioning).
    Returns 4-elem [x1,y1,x2,y2] normalized boxes (GLIGEN's expected format)."""
    k = rng.randint(1, 6)
    boxes, phrases = [], []
    for _ in range(k):
        cls = rng.choice(COCO_NAMES)
        w = rng.uniform(0.08, 0.35)
        h = w * rng.uniform(0.7, 1.4)
        for _ in range(50):
            cx, cy = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
            bx = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            if all(iou4(bx, b) < 0.05 for b in boxes): break
        boxes.append(bx); phrases.append(cls)
    return boxes, phrases

def generate_pool(n_images, out_dir, detector_weights):
    """GLIGEN box-conditioned generation. Pool layout: out/images/, out/labels/.
    Falls back to a real-image stand-in pool if diffusers/GLIGEN is unavailable."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        log(f"loading GLIGEN ({CFG['generation']['model_id']})...")
        pipe = StableDiffusionPipeline.from_pretrained(
            CFG["generation"]["model_id"], torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception as e:
        log(f"GLIGEN unavailable ({e}); falling back to REAL-image stand-in pool (smoke only)")
        return real_standin_pool(n_images, out)
    from ultralytics import YOLO
    det = YOLO(detector_weights)
    rng = random.Random(CFG["generation"]["seed"])
    S = CFG["generation"]["resolution"]
    pool = []
    for i in range(n_images):
        boxes, phrases = layout_sample(rng)
        prompt = "a photo of " + ", ".join(phrases)
        im = pipe(prompt=prompt, negative_prompt="blurry, low quality, deformed",
                  gligen_phrases=phrases, gligen_boxes=boxes,
                  gligen_scheduled_sampling_ratio=0.5,
                  num_inference_steps=20, guidance_scale=7.5).images[0]
        fp = out / "images" / f"img_{i:04d}.jpg"; im.save(fp)
        lines = []
        for b, ph in zip(boxes, phrases):
            x1, y1, x2, y2 = b
            lines.append(f"{NAME2ID[ph]} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
        (out / "labels" / f"img_{i:04d}.txt").write_text("\n".join(lines))
        # box fidelity mu_Q: conditioning boxes (normalized) vs detections (pixels)
        res = det.predict(str(fp), imgsz=DET["imgsz"], conf=0.05, verbose=False)[0]
        # detections are [x1,y1,x2,y2]; prepend dummy cls to match iou() format
        preds = [[0.0] + b.xyxy[0].tolist() for b in res.boxes] if res.boxes is not None else []
        bp = [[0, b[0] * S, b[1] * S, b[2] * S, b[3] * S] for b in boxes]
        ious = [max([iou(b, p) for p in preds] or [0.0]) for b in bp]
        pool.append({"img": str(fp), "mu_q": sum(ious) / len(ious) if ious else 0.0})
        if (i + 1) % 25 == 0: log(f"  generated {i+1}/{n_images}")
    mu_q = sum(p["mu_q"] for p in pool) / len(pool)
    log(f"pool generated: {len(pool)} images, mu_Q={mu_q:.3f}")
    return pool, mu_q

def real_standin_pool(n_images, out_dir):
    """Smoke-mode pool: real images the detector has NOT trained on (poolsrc split),
    annotations known. Falls back to train images only if no poolsrc split exists
    (full-mode GLIGEN-unavailable path) — with the caveat that duplicates make
    U(S) spread degenerate (see build_coco_smoke)."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    src = Path(P["subset_out"]) / "images/poolsrc"
    lab_src = Path(P["subset_out"]) / "labels/poolsrc"
    if not src.exists() or not any(src.glob("*.jpg")):
        src = Path(P["subset_out"]) / "images/train"
        lab_src = Path(P["subset_out"]) / "labels/train"
    imgs = sorted(src.glob("*.jpg"))[:n_images]
    pool = []
    for i, p in enumerate(imgs):
        shutil.copy(p, out / "images" / f"img_{i:04d}.jpg")
        lf = lab_src / p.name.replace(".jpg", ".txt")
        if lf.exists(): shutil.copy(lf, out / "labels" / f"img_{i:04d}.txt")
        pool.append({"img": str(out / "images" / f"img_{i:04d}.jpg"), "mu_q": 1.0})
    log(f"stand-in pool: {len(pool)} held-out real images (mu_Q=1.0 by construction)")
    return pool, 1.0

# --------------------------------------------------------------------------
# 8. Probe subsets: sample -> retrain -> U(S)
# --------------------------------------------------------------------------
def pool_image_features(pool, det):
    """Per-image subset features for pool candidates: failure stats (pool labels as
    GT, baseline detections as predictions) + fidelity mu_Q. Pre-training diagnostics
    only — no synthetic ΔAP anywhere."""
    feats = {}
    for it in pool:
        r = det.predict(it["img"], imgsz=DET["imgsz"], conf=DET["conf"], verbose=False)[0]
        gt = scale_gt_to_pixels(read_labels(label_path_for(it["img"])), it["img"])
        boxes = r.boxes
        # detections are [x1,y1,x2,y2]; prepend dummy cls to match iou() format
        pred = [[0.0] + b.xyxy[0].tolist() for b in boxes] if boxes is not None else []
        confs = [float(b.conf) for b in boxes] if boxes is not None else []
        fs = image_failure(gt, pred, confs, FAIL["match_iou"])
        fs["mu_q"] = it["mu_q"]
        feats[it["img"]] = fs
    return feats

def subset_features(S, feats):
    """Aggregate pool-image features to subset level (μ_FN, μ_FP, μ_Loc, μ_EL2N,
    μ_Q; H_class/H_scale/D_CLIP come in the full protocol)."""
    keys = ["fn_rate", "fp_rate", "loc_err", "conf_entropy", "mu_q"]
    out = {}
    for k in keys:
        vals = [feats[it["img"]][k] for it in S if it["img"] in feats]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out

def build_contrast_subsets(pool, pool_feats, rng):
    """Stratified contrast probes (proposal §10 Stage C: 'Sample probe subsets
    Sᵢ stratified over the feature space'). Ranking the NEUTRAL pool by the
    primary failure feature (μ_FN) and probing the extremes maximizes feature
    variance across subsets — the strongest test of whether failure structure
    predicts U(S) per unit compute (random subsets compress the predictor's
    variance and dilute the signal)."""
    n = len(pool)
    if n < 3:
        return [("ALL", list(pool))]
    size = max(1, n // 3)
    def ranked(key):
        return sorted(pool, key=lambda it: pool_feats[it["img"]].get(key, 0.0), reverse=True)
    return [("HIGH-FN", ranked("fn_rate")[:size]),
            ("LOW-FN", ranked("fn_rate")[-size:]),
            ("MID-FN", ranked("fn_rate")[size:2 * size]),
            ("RAND", rng.sample(pool, size)),
            ("HIGH-FP", ranked("fp_rate")[:size]),
            ("HIGH-LOC", ranked("loc_err")[:size])]

def probe_pipeline(train_paths, root, val_yaml, pool, pool_feats, baseline, cache):
    """For each probe: D_real ∪ S via a YOLO train-txt list, retrain, U(S) = ΔmAP.
    Never mutates D_real or the val set. Times each probe for GPU-h calibration.
    Subsets are stratified contrast probes (see build_contrast_subsets); the
    duplicate-seed noise floor is measured ON the mechanism contrast (HIGH-FN,
    LOW-FN) so the SNR gate directly guards the thesis's key comparison."""
    rng = random.Random(1234)
    subsets = build_contrast_subsets(pool, pool_feats, rng)
    dup_names = {"HIGH-FN", "LOW-FN"}
    dup_seed = DET["seed"] + 1000  # widely separated: seed+1 can be swallowed by deterministic training
    probe_rows = []
    for name, S in subsets:
        agg = subset_features(S, pool_feats)
        seeds = [DET["seed"]] + ([dup_seed] if name in dup_names else [])
        for seed in seeds:
            key = f"probe_{name}_s{seed}"
            if key in cache:
                probe_rows.append(cache[key]); continue
            log(f"probe {name} seed {seed}: {len(S)} pool images + {len(train_paths)} real, training...")
            txt = Path("data") / f"{key}.txt"
            # absolute paths: Ultralytics resolves txt-list entries relative to the
            # dataset path inconsistently across versions; absolute is bulletproof
            img_list = [str(Path(p).resolve()) for p in train_paths]
            img_list += [str(Path(it["img"]).resolve()) for it in S]
            txt.write_text("\n".join(img_list))
            yaml_path = dataset_yaml(root, str(txt), val_yaml, f"{key}.yaml")
            r = train_and_eval(yaml_path, key, DET["epochs_probe"], seed)
            r.update({"U": r["mAP"] - baseline["mAP"], "subset": name, "seed": seed, "n": len(S)})
            r.update(agg)
            probe_rows.append(r); cache[key] = r
            Path(P["results_out"]).mkdir(exist_ok=True)
            Path(P["results_out"], f"probe_results_{config_fingerprint()}.json").write_text(json.dumps(probe_rows))
    # warn if duplicate-seed runs are identical: the noise floor is then unmeasurable
    for name in dup_names:
        pair = [r for r in probe_rows if r["subset"] == name]
        if len(pair) == 2 and pair[0]["mAP"] == pair[1]["mAP"]:
            log(f"WARNING: duplicate seeds for {name} produced identical mAP "
                f"({pair[0]['mAP']:.4f}) — Ultralytics seed override may not be taking "
                f"effect; noise floor is below measurement precision")
    return probe_rows

# --------------------------------------------------------------------------
# 9. Report + verdict
# --------------------------------------------------------------------------
def report(probe_rows, corr_fd, mu_q, hours_per_probe, baseline, mode="smoke",
           corr_measured=False):
    print("\n" + "=" * 78)
    print("PREDU-OD PILOT REPORT")
    print("=" * 78)
    Us = [r["U"] for r in probe_rows if r["seed"] == DET["seed"]]
    spread = max(Us) - min(Us)
    mean_u = sum(Us) / len(Us)
    std = (sum((u - mean_u) ** 2 for u in Us) / len(Us)) ** 0.5
    dups = [abs(r1["U"] - r2["U"]) for r1 in probe_rows for r2 in probe_rows
            if r1["subset"] == r2["subset"] and r1["seed"] != r2["seed"]]
    noise = (sum(d * d for d in dups) / max(1, len(dups))) ** 0.5 if dups else 0.0
    # no signal (flat U) or no noise measurement => SNR is 0, not inf (honest guard)
    if noise > 1e-4 and std > 1e-6:
        snr = std / noise
    elif std > 1e-4:
        snr = None  # spread exists but noise floor unmeasurable -> SNR undefined, gate SKIPs
    else:
        snr = 0.0   # genuinely flat U -> no signal
    snr_str = "inf/unmeasurable" if snr is None else f"{snr:.3f}"
    print(f"\n[1] GLIGEN box fidelity mu_Q            : {mu_q:.3f}  (gate >= {GATES['box_fidelity_min']})")
    print(f"[2] Baseline mAP@[.5:.95] (10% regime)  : {baseline['mAP']:.3f}")
    print(f"    corr(F, D) (max |spearman|)         : {corr_fd:.3f}  (warn >= {GATES['corr_fd_warn']})")
    print(f"[3] Probe spread of U(S) (n={len(Us)})  : {spread:.3f} (std {std:.3f}, mean {mean_u:+.3f})")
    print(f"    2-seed noise floor (n={len(dups)})  : {noise:.4f}")
    print(f"    SNR = std/noise                     : {snr_str}  (gate >= {GATES['snr_min']})")
    feats = {}
    for r in probe_rows:
        if r["seed"] != DET["seed"]: continue
        for f in ("fn_rate", "fp_rate", "loc_err", "conf_entropy", "mu_q"):
            feats.setdefault(f, []).append(r.get(f, 0.0))
    taus = {f: kendall_tau(v, Us) for f, v in feats.items() if len(v) == len(Us)}
    best_f, best_t = max(taus.items(), key=lambda kv: abs(kv[1]))
    print(f"    preliminary Kendall-tau best feature: {best_f} -> {best_t:.3f}  (gate >= {GATES['prelim_tau_min']})")
    # mechanism contrast: the thesis's headline comparison (pre-registered DMR-adjacent)
    def row_by(name):
        got = [r for r in probe_rows if r["subset"] == name and r["seed"] == DET["seed"]]
        return got[0]["U"] if got else None
    u_hi, u_lo = row_by("HIGH-FN"), row_by("LOW-FN")
    if u_hi is not None and u_lo is not None:
        du = u_hi - u_lo
        hi_dups = [abs(r1["U"] - r2["U"]) for r1 in probe_rows for r2 in probe_rows
                   if r1["subset"] == "HIGH-FN" and r2["subset"] == "HIGH-FN" and r1["seed"] != r2["seed"]]
        lo_dups = [abs(r1["U"] - r2["U"]) for r1 in probe_rows for r2 in probe_rows
                   if r1["subset"] == "LOW-FN" and r2["subset"] == "LOW-FN" and r1["seed"] != r2["seed"]]
        dup_noise = (sum(hi_dups) + sum(lo_dups)) / max(1, len(hi_dups) + len(lo_dups))
        print(f"[4] CONTRAST dU = U(HIGH-FN) - U(LOW-FN) : {du:+.4f}  (dup-seed noise {dup_noise:.4f})")
        if abs(dup_noise) > 1e-6 and abs(du) / max(1e-6, dup_noise) >= 1.0:
            print(f"    -> failure-aligned subsets differ beyond the noise floor (mechanism signal)")
        else:
            print(f"    -> contrast within noise floor (mechanism not yet detectable at this scale)")
    print(f"[5] GPU-hours per probe train           : {hours_per_probe:.2f}  (gate <= {GATES['gpu_hours_per_probe_max']})")
    print(f"    projected n_probes @60 GPU-h        : {int(60 / max(1e-6, hours_per_probe))}")
    print("\n" + "-" * 78)
    checks = [
        ("GLIGEN mu_Q >= 0.55", mu_q >= GATES["box_fidelity_min"]),
        # snr=None (unmeasurable noise floor) must not crash the comparison;
        # the gate is SKIPped below, so the ok value is unused for that case
        ("SNR >= 1.0 (probe signal > noise)", snr is not None and snr >= GATES["snr_min"]),
        ("prelim tau >= 0.3", abs(best_t) >= GATES["prelim_tau_min"]),
        ("corr(F,D) < 0.9", corr_fd < GATES["corr_fd_warn"]),
        ("GPU-h/probe <= 0.6", hours_per_probe <= GATES["gpu_hours_per_probe_max"]),
    ]
    skipped = set()
    for name, ok in checks:
        if mode == "smoke" and name.startswith("GLIGEN"):
            skipped.add(name)
            print(f"  [SKIP] {name}  (stand-in pool in smoke mode; full mode required)")
        elif not corr_measured and name.startswith("corr(F,D)"):
            skipped.add(name)
            print(f"  [SKIP] {name}  (EL2N difficulty not run in --quick; measured in full protocol)")
        elif name.startswith("SNR") and snr is None:
            skipped.add(name)
            print(f"  [SKIP] {name}  (probe spread exists but duplicate-seed noise floor is unmeasurable — see WARNING)")
        else:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    go = all(ok for name, ok in checks if name not in skipped)
    print("-" * 78)
    print(f"VERDICT: {'GO — proceed to full 64-80 probe protocol' if go else 'NO-GO — see §13 mitigations (shrink probes / scope, or pivot)'}")
    print("=" * 78)
    return go

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "smoke"], default="smoke")
    ap.add_argument("--quick", action="store_true", help="reduced epochs/subsets for a fast run")
    ap.add_argument("--force", action="store_true", help="ignore caches, recompute")
    ap.add_argument("--frac", type=float, default=None,
                    help="COCO train fraction for D_real (0.02 = 2%%; default from config 0.1)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="equal epochs for baseline AND probes (fair dU; default 60/30)")
    args = ap.parse_args()
    if args.frac is not None:
        CFG["coco_subset"]["fraction"] = args.frac
        log(f"regime override: {args.frac:.0%} of COCO train2017 (data-sparse => max headroom)")
    if args.epochs is not None:
        DET.update(epochs_baseline=args.epochs, epochs_probe=args.epochs)
        log(f"equal-schedule override: {args.epochs} epochs (baseline = probes)")
    global CURRENT_MODE
    CURRENT_MODE = args.mode
    log(f"pilot_probe_signal.py version {SCRIPT_VERSION}")
    Path(P["results_out"]).mkdir(parents=True, exist_ok=True)
    if args.quick:
        DET.update(epochs_baseline=20, epochs_probe=20, batch=8)  # equal schedule: U = ΔmAP at same training effort
        PROB["n_subsets"] = 6
        if args.mode == "smoke":
            # subsets must be SMALLER than the 40-image stand-in pool, otherwise
            # every probe trains on the identical image set -> spread 0 -> false NO-GO
            PROB["sizes"] = [30]
        else:
            # full mode: GLIGEN pool has n_images candidates; keep realistic subset
            # sizes so probe ΔmAP reflects actual marginal utility
            PROB["sizes"] = [min(100, CFG["generation"]["n_images"]),
                              min(150, CFG["generation"]["n_images"])]

    # probe cache is fingerprinted by config+version so stale rows from older
    # runs (different epochs/pool/subset sizes) are never reused silently
    cache = {}
    cpath = Path(P["results_out"], f"probe_results_{config_fingerprint()}.json")
    if cpath.exists() and not args.force:
        cache = {f"probe_{r['subset']}_s{r['seed']}": r for r in json.loads(cpath.read_text())}
        if cache:
            log(f"probe cache hit ({len(cache)} rows, fingerprint {config_fingerprint()})")

    log("step 1/6: COCO setup")
    if args.mode == "smoke":
        train_paths, val_paths, root = build_coco_smoke()
    else:
        train_paths, val_paths, root = build_coco_full()
    tdir, vdir = root / "images/train", root / "images/val"
    train_yaml = dataset_yaml(root, str(tdir), str(vdir), "train.yaml")

    log("step 2/6: baseline YOLO11s (early checkpoints for difficulty)")
    baseline = train_and_eval(train_yaml, "baseline", DET["epochs_baseline"], DET["seed"], save_period=1)
    log(f"baseline mAP@[.5:.95] = {baseline['mAP']:.3f} ({baseline['hours']:.2f} GPU-h)")
    Path(P["results_out"], "baseline.json").write_text(json.dumps(baseline))

    log("step 3/6: early-training difficulty + failure profile")
    diff = early_difficulty(Path(baseline["save_dir"]), train_paths) if not args.quick else {}
    from ultralytics import YOLO
    det = YOLO(baseline["best"])
    rows = []
    for p in train_paths[:400]:
        r = det.predict(p, imgsz=DET["imgsz"], conf=DET["conf"], verbose=False)[0]
        gt = scale_gt_to_pixels(read_labels(label_path_for(p)), p)
        boxes = r.boxes
        # detections are [x1,y1,x2,y2]; prepend dummy cls to match iou() format
        pred = [[0.0] + b.xyxy[0].tolist() for b in boxes] if boxes is not None else []
        confs = [float(b.conf) for b in boxes] if boxes is not None else []
        rows.append((p, image_failure(gt, pred, confs, FAIL["match_iou"])))
    corr_fd = 0.0
    if diff:
        keys = [p for p, _ in rows if p in diff]
        if len(keys) >= 8:
            c = [spearman([dict(rows)[k][f] for k in keys], [diff[k] for k in keys]) for f in
                 ("fn_rate", "fp_rate", "loc_err", "conf_entropy")]
            corr_fd = max(abs(v) for v in c)
    log(f"corr(F,D) = {corr_fd:.3f}")

    log("step 4/6: candidate pool (GLIGEN or stand-in)")
    if args.mode == "full":
        pool, mu_q = generate_pool(CFG["generation"]["n_images"], P["pool_out"], baseline["best"])
    else:
        pool, mu_q = real_standin_pool(60, P["pool_out"])
    log("step 4b/6: pool-image features (failure stats + fidelity)")
    pool_feats = pool_image_features(pool, det)

    log("step 5/6: probe subsets -> retrain -> U(S)")
    probe_rows = probe_pipeline(train_paths, root, str(vdir), pool, pool_feats, baseline, cache)

    log("step 6/6: calibration report")
    hours_per_probe = sum(r["hours"] for r in probe_rows) / max(1, len(probe_rows))
    report(probe_rows, corr_fd, mu_q, hours_per_probe, baseline,
           mode=args.mode, corr_measured=bool(diff))

if __name__ == "__main__":
    main()
