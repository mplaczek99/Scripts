#!/usr/bin/env python3
# match_frames.py
# Hard-coded for ClipA.mp4 and ClipB.mp4
# Saves best 10 frame matches into ./matched_frames
# Compact console output for ~80x24 terminals
# Only CLI flag: --model (dinov2 or resnet50)

import os, sys, math, heapq, shutil, argparse
import cv2
import numpy as np

# -------------------- CLI args (only --model) --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Frame matcher (compact TTY output)")
    p.add_argument("--model", default="dinov2", choices=["dinov2", "resnet50"],
                   help="Feature extractor backbone")
    return p.parse_args()

ARGS = parse_args()
MODEL_NAME = ARGS.model

# -------------------- Tunables --------------------
VIDEO_A = "ClipA.mp4"
VIDEO_B = "ClipB.mp4"
TOPK = 10
STRIDE = 1
OUTDIR = "matched_frames"

# -------------------- CUDA allocator hint --------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

# -------------------- Optional PyTorch/TIMM stack --------------------
TORCH_OK = True
TORCH_IMPORT_ERR = None
try:
    import torch
    import torch.nn.functional as F
    import timm
    from timm.data import resolve_model_data_config
except Exception as e:
    TORCH_OK = False
    TORCH_IMPORT_ERR = e

# -------------------- Pretty printing helpers (compact) --------------------
def term_cols():
    try:
        return max(60, min(80, shutil.get_terminal_size((80, 24)).columns))
    except Exception:
        return 80

def sep(char="="):
    print(char * term_cols())

def kv(label: str, value: str):
    print(f"{label}: {value}")

def truncate(s: str, maxw: int) -> str:
    if len(s) <= maxw: return s
    if maxw <= 1: return "…"
    return s[:maxw-1] + "…"

# -------------------- Search defaults --------------------
A_BLOCK_INIT = 2048
B_BLOCK_INIT = 2048
SUBBLOCK_TOP = 4000
GLOBAL_HEAP_CAP = 80000

def get_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return 30.0 if (not fps or math.isnan(fps) or fps <= 0) else float(fps)

def load_feat_extractor(model_name: str, device_pref: str = "auto"):
    if not TORCH_OK:
        raise RuntimeError("PyTorch/TIMM not available. Install torch, timm, etc.")
    dev = "cuda" if (device_pref != "cpu" and torch.cuda.is_available()) else "cpu"

    if model_name.lower() == "dinov2":
        backbone = "vit_base_patch14_dinov2"
        model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="token")
    elif model_name.lower() == "resnet50":
        backbone = "resnet50"
        model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
    else:
        raise ValueError("Unknown model. Use dinov2 or resnet50.")

    model.eval()
    if dev == "cuda":
        try: torch.backends.cudnn.benchmark = True
        except Exception: pass
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
        model = model.to(memory_format=torch.channels_last).to(dev)
    else:
        model = model.to(dev)

    data_cfg = resolve_model_data_config(model)
    H, W = int(data_cfg["input_size"][1]), int(data_cfg["input_size"][2])
    mean = torch.tensor(data_cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor(data_cfg["std"],  dtype=torch.float32).view(1, 3, 1, 1)
    return model, (H, W), mean, std, dev

def _noop_dec(fn): return fn
_no_grad = getattr(torch, "no_grad", _noop_dec) if TORCH_OK else _noop_dec

def _autocast_ctx(device: str, dtype):
    if not TORCH_OK or device != "cuda":
        class _NoCtx:
            def __enter__(self): return None
            def __exit__(self, *exc): return False
        return _NoCtx()
    if hasattr(torch, "autocast"):
        return torch.autocast("cuda", dtype=dtype)
    from torch.cuda.amp import autocast
    return autocast(dtype=dtype)

@_no_grad
def _forward_batch_gpu(frames_cpu, model, input_hw, mean, std, device, amp_dtype):
    x = torch.from_numpy(np.stack(frames_cpu, axis=0))
    x = x.to(device, non_blocking=True).permute(0,3,1,2).contiguous(memory_format=torch.channels_last)
    x = x.to(torch.float32).div_(255.0)
    x = torch.nn.functional.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
    x = (x - mean.to(device, non_blocking=True)) / std.to(device, non_blocking=True)
    with _autocast_ctx(device, amp_dtype):
        feats = model(x)
    return feats

@_no_grad
def embed_video(video_path: str, model, input_hw, mean, std, device,
                stride: int = 1, init_batch: int = 24, min_batch: int = 2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    amp_dtype = torch.float16 if (TORCH_OK and device == "cuda") else torch.bfloat16
    feats_chunks, buf_frames = [], []
    idx, batch = 0, init_batch

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buf_frames.append(frame_rgb)
        if len(buf_frames) >= batch:
            feats = _forward_batch_gpu(buf_frames[:batch], model, input_hw, mean, std, device, amp_dtype)
            feats_chunks.append(feats.detach().cpu())
            del feats, buf_frames[:batch]
            if device == "cuda": torch.cuda.empty_cache()
        idx += 1

    while buf_frames:
        take = min(batch, len(buf_frames))
        feats = _forward_batch_gpu(buf_frames[:take], model, input_hw, mean, std, device, amp_dtype)
        feats_chunks.append(feats.detach().cpu())
        del feats, buf_frames[:take]
        if device == "cuda": torch.cuda.empty_cache()

    cap.release()
    if not feats_chunks:
        dim = getattr(model, "num_features", 0) if TORCH_OK else 0
        return np.zeros((0, dim), np.float32)
    return torch.cat(feats_chunks, dim=0).numpy().astype(np.float32)

def l2norm_torch(x): return F.normalize(x, p=2, dim=1)

@_no_grad
def top_candidates_gpu(feats_a, feats_b, device, keep_cap=GLOBAL_HEAP_CAP,
                       subblock_top=SUBBLOCK_TOP, a_block_init=A_BLOCK_INIT, b_block_init=B_BLOCK_INIT):
    if not TORCH_OK: raise RuntimeError("PyTorch required.")
    aT = l2norm_torch(torch.from_numpy(feats_a).to(device, non_blocking=True))
    bT = l2norm_torch(torch.from_numpy(feats_b).to(device, non_blocking=True))
    Na, _ = aT.shape; Nb, _ = bT.shape
    heap, push, pushpop = [], heapq.heappush, heapq.heappushpop
    for ai in range(0, Na, a_block_init):
        a_blk = aT[ai:ai+a_block_init]
        for bj in range(0, Nb, b_block_init):
            b_blk = bT[bj:bj+b_block_init]
            sims = a_blk @ b_blk.t()
            k = min(subblock_top, sims.numel())
            if k > 0:
                vals, flat_idx = torch.topk(sims.view(-1), k, largest=True, sorted=True)
                rows = (flat_idx // b_blk.shape[0]).tolist()
                cols = (flat_idx %  b_blk.shape[0]).tolist()
                for s, r, c in zip(vals.tolist(), rows, cols):
                    ia, jb = ai + r, bj + c
                    if len(heap) < keep_cap: push(heap, (s, ia, jb))
                    elif s > heap[0][0]: pushpop(heap, (s, ia, jb))
            del sims
            if device == "cuda": torch.cuda.empty_cache()
    heap.sort(key=lambda x: x[0], reverse=True)
    return [(ia, jb, float(s)) for (s, ia, jb) in heap]

def time_from_index(video_path, frame_index, stride):
    return (frame_index * stride) / max(get_fps(video_path), 1e-6)

def save_frame_at_time(video_path, t_sec, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t_sec) * 1000.0)
    ok, frame_bgr = cap.read(); cap.release()
    if not ok or frame_bgr is None: return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return cv2.imwrite(out_path, frame_bgr)

def fmt_t_short(t: float) -> str:
    # 6-wide with 3 decimals, zero-padded: e.g., 030.467
    return f"{t:06.3f}"

def build_filename(rank: int, which: str, t: float) -> str:
    # which="A" or "B"; filename like 01_A_030.467.png
    return f"{rank:02d}_{which}_{fmt_t_short(t)}.png"

def print_results_table(rows):
    """
    rows: list of dicts with keys rank, pct, tA, tB, fileA, fileB
    Fits within terminal width by truncating file columns if needed.
    """
    if not rows:
        print("(no rows)")
        return

    # Base column definitions
    cols = [
        ("Rk",  lambda r: f"{r['rank']:>2}", 2),
        ("Match%", lambda r: f"{r['pct']:>6.2f}%", 7),
        ("tA(s)", lambda r: f"{r['tA']:>7.3f}", 7),
        ("tB(s)", lambda r: f"{r['tB']:>7.3f}", 7),
        ("A file", lambda r: r["fileA"], 14),  # min; will expand/truncate
        ("B file", lambda r: r["fileB"], 14),
    ]

    # Compute widths with a cap for terminal width
    # Start with fixed widths for first 4 columns (+ 5 spaces between 6 columns)
    fixed_sum = sum(w for _,_,w in cols[:4]) + 5  # spaces between columns
    # Distribute remaining space to A/B file columns equally
    total = term_cols()
    rem = max(20, total - fixed_sum)  # ensure some space
    each = rem // 2
    widths = [cols[i][2] for i in range(4)] + [each, rem - each]

    # Print header
    header_parts = []
    for (title, _, _), w in zip(cols, widths):
        header_parts.append(truncate(title.ljust(w), w))
    print(" ".join(header_parts))
    print(" ".join(("─"*w) for w in widths))

    # Rows
    for r in rows:
        parts = []
        for (title, fn, _), w in zip(cols, widths):
            val = fn(r)
            # Truncate filenames; other columns should already fit
            parts.append(truncate(val.rjust(w) if title in ("Rk","Match%","tA(s)","tB(s)") else val.ljust(w), w))
        print(" ".join(parts))

def main():
    sep("=")
    print("Frame Match (GPU tiled)")
    sep("-")
    kv("Video A", VIDEO_A)
    kv("Video B", VIDEO_B)
    kv("Model", MODEL_NAME)
    kv("Output", OUTDIR)
    kv("TopK", str(TOPK))
    kv("Stride", str(STRIDE))
    sep("=")

    os.makedirs(OUTDIR, exist_ok=True)
    print("[1/3] Loading model…")
    model, input_hw, mean, std, device = load_feat_extractor(MODEL_NAME, "auto")
    print(f"    Ready on {device} (input {input_hw[0]}x{input_hw[1]})")

    print("[2/3] Embedding videos…")
    feats_a = embed_video(VIDEO_A, model, input_hw, mean, std, device,
                          stride=STRIDE, init_batch=(12 if MODEL_NAME=="dinov2" else 64))
    feats_b = embed_video(VIDEO_B, model, input_hw, mean, std, device,
                          stride=STRIDE, init_batch=(12 if MODEL_NAME=="dinov2" else 64))
    print(f"    A frames: {feats_a.shape[0]}")
    print(f"    B frames: {feats_b.shape[0]}")
    if feats_a.shape[0] == 0 or feats_b.shape[0] == 0:
        print("No frames sampled; aborting.")
        sys.exit(1)

    print("[3/3] Similarity search…")
    candidates = top_candidates_gpu(feats_a, feats_b, device,
                                    keep_cap=max(GLOBAL_HEAP_CAP, TOPK*8000),
                                    subblock_top=SUBBLOCK_TOP,
                                    a_block_init=A_BLOCK_INIT, b_block_init=B_BLOCK_INIT)
    if not candidates:
        print("No matches found.")
        sys.exit(2)

    selected = candidates[:TOPK]
    rows = []
    for rank, (ia, ib, s) in enumerate(selected, 1):
        ta, tb = time_from_index(VIDEO_A, ia, STRIDE), time_from_index(VIDEO_B, ib, STRIDE)
        fileA = build_filename(rank, "A", ta)
        fileB = build_filename(rank, "B", tb)
        save_frame_at_time(VIDEO_A, ta, os.path.join(OUTDIR, fileA))
        save_frame_at_time(VIDEO_B, tb, os.path.join(OUTDIR, fileB))
        rows.append({
            "rank": rank,
            "pct": s * 100.0,
            "tA": ta,
            "tB": tb,
            "fileA": fileA,
            "fileB": fileB
        })

    sep("=")
    print("Top Matches")
    sep("-")
    print_results_table(rows)
    sep("=")
    print(f"Saved PNGs -> {OUTDIR}/")

if __name__ == "__main__":
    main()

