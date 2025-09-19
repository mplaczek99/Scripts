#!/usr/bin/env python3
# match_frames_pretty.py
# Hard-coded for ClipA.mp4 and ClipB.mp4
# Saves best 10 frame matches into ./matched_frames
# Pretty console output with banners + final results table

import os, sys, math, heapq, shutil
import cv2
import numpy as np

# -------------------- Tunables --------------------
VIDEO_A = "ClipA.mp4"
VIDEO_B = "ClipB.mp4"
TOPK = 10             # how many best pairs to save
STRIDE = 1            # sample every Nth frame (1=native)
MODEL_NAME = "dinov2" # or "resnet50"
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

# -------------------- Pretty printing helpers --------------------
def _supports_color() -> bool:
    try:
        return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb", "")
    except Exception:
        return False

_COLOR = _supports_color()

def _c(code: str) -> str:
    return f"\033[{code}m" if _COLOR else ""

def C_RESET(): return _c("0")
def C_BOLD():  return _c("1")
def C_DIM():   return _c("2")
def C_CYAN():  return _c("36")
def C_GREEN(): return _c("32")
def C_YELLOW():return _c("33")
def C_MAG():   return _c("35")
def C_BLUE():  return _c("34")

def banner(title: str):
    cols = max(60, shutil.get_terminal_size((80, 20)).columns)
    line = "─" * (cols - 2)
    print(f"{_c('1;36')}┌{line}┐{C_RESET()}")
    center = title.center(cols - 2)
    print(f"{_c('1;36')}│{C_RESET()}{_c('1')}{center}{C_RESET()}{_c('1;36')}│{C_RESET()}")
    print(f"{_c('1;36')}└{line}┘{C_RESET()}")

def kv(label: str, value: str):
    print(f"{C_DIM()}{label:>14}{C_RESET()}: {value}")

def ok(msg: str):
    print(f"{C_GREEN()}✓{C_RESET()} {msg}")

def warn(msg: str):
    print(f"{C_YELLOW()}!{C_RESET()} {msg}")

def log(msg: str):
    print(msg, flush=True)

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
        raise RuntimeError(
            "PyTorch/TIMM not available. Install:\n"
            "  pip install --upgrade torch torchvision timm pillow\n"
            f"Import error: {TORCH_IMPORT_ERR}"
        )
    dev = "cuda" if (device_pref != "cpu" and torch.cuda.is_available()) else "cpu"

    if model_name.lower() in ("dinov2", "vitb14", "dino"):
        backbone = "vit_base_patch14_dinov2"
        model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="token")
    elif model_name.lower() in ("resnet50", "rn50", "resnet"):
        backbone = "resnet50"
        model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
    else:
        raise ValueError("Unknown model. Use 'dinov2' or 'resnet50'.")

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
    x = torch.from_numpy(np.stack(frames_cpu, axis=0))           # [N,H,W,3] uint8
    x = x.to(device, non_blocking=True).permute(0,3,1,2).contiguous(memory_format=torch.channels_last)
    x = x.to(torch.float32).div_(255.0)
    x = torch.nn.functional.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
    x = (x - mean.to(device, non_blocking=True)) / std.to(device, non_blocking=True)
    with _autocast_ctx(device, amp_dtype):
        feats = model(x)
    return feats  # [N, D] on device

@_no_grad
def embed_video(video_path: str, model, input_hw, mean, std, device,
                stride: int = 1, init_batch: int = 24, min_batch: int = 2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    amp_dtype = torch.float16 if (TORCH_OK and device == "cuda") else torch.bfloat16
    feats_chunks, buf_frames, buf_ts = [], [], []
    idx, batch = 0, init_batch

    # light inline progress (frame count every ~1000 samples)
    sampled = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buf_frames.append(frame_rgb)
            buf_ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            sampled += 1
            if sampled % 1000 == 0:
                print(f"{C_DIM()}  …sampled {sampled} frames{C_RESET()}", end="\r", flush=True)

        if len(buf_frames) >= batch:
            while True:
                try:
                    feats = _forward_batch_gpu(buf_frames[:batch], model, input_hw, mean, std, device, amp_dtype)
                    feats_chunks.append(feats.detach().cpu())
                    del feats, buf_frames[:batch], buf_ts[:batch]
                    if device == "cuda": torch.cuda.empty_cache()
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and batch > min_batch:
                        batch = max(min_batch, batch // 2)
                        if device == "cuda": torch.cuda.empty_cache()
                        continue
                    raise
        idx += 1

    # tail
    while buf_frames:
        take = min(batch, len(buf_frames))
        feats = _forward_batch_gpu(buf_frames[:take], model, input_hw, mean, std, device, amp_dtype)
        feats_chunks.append(feats.detach().cpu())
        del feats, buf_frames[:take], buf_ts[:take]
        if device == "cuda": torch.cuda.empty_cache()

    cap.release()
    if feats_chunks:
        print(" " * 40, end="\r")  # clear progress line
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
    Ablk, Bblk = a_block_init, b_block_init

    for ai in range(0, Na, Ablk):
        a_blk = aT[ai:ai+Ablk]
        for bj in range(0, Nb, Bblk):
            b_blk = bT[bj:bj+Bblk]
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

def print_results_table(rows):
    """
    rows: list of dicts with keys rank, cos, tA, tB, fileA, fileB
    """
    cols = [("Rank","#",4), ("Cosine","cos",8), ("A time (s)","tA",11), ("B time (s)","tB",11),
            ("A file","fileA",26), ("B file","fileB",26)]
    # compute widths
    widths = []
    for title, key, minw in cols:
        w = max(minw, len(title), max((len(str(r[key])) for r in rows), default=0))
        widths.append(w)

    # header
    hdr = "  " + "  ".join(f"{C_BOLD()}{title:<{w}}{C_RESET()}" for (title,_,_), w in zip(cols, widths))
    sep = "  " + "  ".join("─"*w for w in widths)
    print(hdr)
    print(sep)
    # rows
    for r in rows:
        line = "  " + "  ".join([
            f"{r['rank']:<{widths[0]}}",
            f"{r['cos']:<{widths[1]}.6f}",
            f"{r['tA']:<{widths[2]}.3f}",
            f"{r['tB']:<{widths[3]}.3f}",
            f"{r['fileA']:<{widths[4]}}",
            f"{r['fileB']:<{widths[5]}}",
        ])
        print(line)

def main():
    banner("Frame Matching (DINO/ResNet, GPU tiled search)")
    kv("Video A", VIDEO_A)
    kv("Video B", VIDEO_B)
    kv("Model", MODEL_NAME)
    kv("Output", OUTDIR)
    kv("TopK", str(TOPK))
    kv("Stride", str(STRIDE))
    print()

    os.makedirs(OUTDIR, exist_ok=True)

    # Load model
    print(f"{C_CYAN()}[1/3]{C_RESET()} Loading model…")
    model, input_hw, mean, std, device = load_feat_extractor(MODEL_NAME, "auto")
    ok(f"Model on {device} (input {input_hw[0]}×{input_hw[1]})")

    # Embed A
    print(f"{C_CYAN()}[2/3]{C_RESET()} Embedding videos…")
    fps_a, fps_b = get_fps(VIDEO_A), get_fps(VIDEO_B)
    kv("FPS A", f"{fps_a:.3f}")
    kv("FPS B", f"{fps_b:.3f}")
    init_batch = 12 if MODEL_NAME == "dinov2" else 64

    feats_a = embed_video(VIDEO_A, model, input_hw, mean, std, device,
                          stride=STRIDE, init_batch=init_batch)
    feats_b = embed_video(VIDEO_B, model, input_hw, mean, std, device,
                          stride=STRIDE, init_batch=init_batch)
    ok(f"Embedded A: {feats_a.shape[0]} frames, dim {feats_a.shape[1] if feats_a.size else 0}")
    ok(f"Embedded B: {feats_b.shape[0]} frames, dim {feats_b.shape[1] if feats_b.size else 0}")
    if feats_a.shape[0] == 0 or feats_b.shape[0] == 0:
        warn("No frames sampled; aborting.")
        sys.exit(1)

    # Similarity search
    print(f"{C_CYAN()}[3/3]{C_RESET()} Searching similarities (tiled)…")
    candidates = top_candidates_gpu(
        feats_a, feats_b, device,
        keep_cap=max(GLOBAL_HEAP_CAP, TOPK * 8000),
        subblock_top=SUBBLOCK_TOP,
        a_block_init=A_BLOCK_INIT, b_block_init=B_BLOCK_INIT
    )
    if not candidates:
        warn("No similarity candidates found.")
        sys.exit(2)

    # Take top-K raw matches
    selected = candidates[:TOPK]

    # Save + collect pretty table rows
    rows = []
    for rank, (ia, ib, s) in enumerate(selected, 1):
        ta, tb = time_from_index(VIDEO_A, ia, STRIDE), time_from_index(VIDEO_B, ib, STRIDE)
        fileA = f"{rank:04d}_A_{ta:09.3f}s.png"
        fileB = f"{rank:04d}_B_{tb:09.3f}s.png"
        a_path = os.path.join(OUTDIR, fileA)
        b_path = os.path.join(OUTDIR, fileB)

        ok_a = save_frame_at_time(VIDEO_A, ta, a_path)
        ok_b = save_frame_at_time(VIDEO_B, tb, b_path)
        if not (ok_a and ok_b):
            warn(f"Save failed for rank {rank}: A={ok_a}, B={ok_b}")
        rows.append({
            "rank": rank,
            "cos": s,
            "tA": ta,
            "tB": tb,
            "fileA": fileA,
            "fileB": fileB,
        })

    print()
    banner("Top Matches")
    print_results_table(rows)
    print()
    ok(f"Saved PNGs → {C_BOLD()}{OUTDIR}{C_RESET()}")
    print(f"{C_DIM()}(Tip: open both files per row side-by-side to visually compare){C_RESET()}")

if __name__ == "__main__":
    main()

