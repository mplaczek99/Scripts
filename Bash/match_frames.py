#!/usr/bin/env python3
# match_frames.py
import os, sys, math, argparse, heapq
import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import timm
    from timm.data import resolve_model_data_config
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    TORCH_IMPORT_ERR = e

# -------------------- GPU Search Tiling (safe defaults for 12GB) --------------------
A_BLOCK_INIT = 2048
B_BLOCK_INIT = 2048
SUBBLOCK_TOP = 4000           # top-k kept per (A-block × B-block)
GLOBAL_HEAP_CAP = 80000       # global cap before NMS (increase for more thorough search)

def log(m): print(m, flush=True)

def seconds_to_hms(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:d}:{m:02d}:{s:06.3f}" if h > 0 else f"{m:d}:{s:06.3f}"

def load_feat_extractor(model_name: str, device_pref: str):
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
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
        except Exception: pass
        model = model.to(memory_format=torch.channels_last).to(dev)
    else:
        model = model.to(dev)

    data_cfg = resolve_model_data_config(model)
    H, W = data_cfg["input_size"][1], data_cfg["input_size"][2]
    mean = torch.tensor(data_cfg["mean"]).view(1, 3, 1, 1)
    std  = torch.tensor(data_cfg["std"]).view(1, 3, 1, 1)
    return model, (H, W), mean, std, dev

def get_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return 30.0 if (not fps or math.isnan(fps) or fps <= 0) else float(fps)

@torch.no_grad()
def _forward_batch_gpu(frames_cpu, model, input_hw, mean, std, device, amp_dtype):
    """
    frames_cpu: list of RGB uint8 arrays [H,W,3]
    Returns feats on *device*.
    """
    x = torch.from_numpy(np.stack(frames_cpu, axis=0))           # [N,H,W,3] uint8
    x = x.to(device, non_blocking=True).permute(0,3,1,2).contiguous(memory_format=torch.channels_last)
    x = x.to(torch.float32).div_(255.0)
    x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
    x = (x - mean.to(device)) / std.to(device)
    if device == "cuda":
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            feats = model(x)
    else:
        feats = model(x)
    return feats  # [N, D] on device

@torch.no_grad()
def embed_video(video_path: str, model, input_hw, mean, std, device,
                stride: int = 1, init_batch: int = 24, min_batch: int = 2):
    """
    OOM-safe embedding with batch downshift. Moves each chunk to CPU immediately.
    For DINOv2 (518×518), start smaller (e.g., 12–24). ResNet50 can go much larger.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    feats_chunks = []
    buf_frames, buf_ts = [], []
    idx, batch = 0, init_batch

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buf_frames.append(frame_rgb)
            buf_ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        if len(buf_frames) >= batch:
            while True:
                try:
                    feats = _forward_batch_gpu(buf_frames[:batch], model, input_hw, mean, std, device, amp_dtype)
                    feats_chunks.append(feats.detach().cpu())  # <-- free VRAM immediately
                    del feats
                    del buf_frames[:batch]; del buf_ts[:batch]
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
        while True:
            try:
                take = min(batch, len(buf_frames))
                feats = _forward_batch_gpu(buf_frames[:take], model, input_hw, mean, std, device, amp_dtype)
                feats_chunks.append(feats.detach().cpu())
                del feats
                del buf_frames[:take]; del buf_ts[:take]
                if device == "cuda": torch.cuda.empty_cache()
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch > min_batch:
                    batch = max(min_batch, batch // 2)
                    if device == "cuda": torch.cuda.empty_cache()
                    continue
                raise

    cap.release()
    if not feats_chunks:
        return np.zeros((0, getattr(model, "num_features", 0)), np.float32)

    feats = torch.cat(feats_chunks, dim=0).numpy().astype(np.float32)  # CPU
    return feats

def l2norm_torch(x): return F.normalize(x, p=2, dim=1)

@torch.no_grad()
def top_candidates_gpu(feats_a: np.ndarray, feats_b: np.ndarray, device: str,
                       keep_cap=GLOBAL_HEAP_CAP, subblock_top=SUBBLOCK_TOP,
                       a_block_init=A_BLOCK_INIT, b_block_init=B_BLOCK_INIT):
    """
    GPU cosine similarities in tiles with OOM backoff. Returns list[(i, j, sim)].
    """
    aT = torch.from_numpy(feats_a).to(device, non_blocking=True)
    bT = torch.from_numpy(feats_b).to(device, non_blocking=True)
    aT = l2norm_torch(aT)
    bT = l2norm_torch(bT)
    Na, D = aT.shape; Nb, _ = bT.shape

    heap = []
    push, pushpop = heapq.heappush, heapq.heappushpop
    Ablk, Bblk = a_block_init, b_block_init

    while True:
        try:
            for ai in range(0, Na, Ablk):
                a_blk = aT[ai:ai+Ablk]
                Pa = a_blk.shape[0]
                for bj in range(0, Nb, Bblk):
                    b_blk = bT[bj:bj+Bblk]
                    Pb = b_blk.shape[0]

                    sims = a_blk @ b_blk.t()  # [Pa, Pb]

                    k = min(subblock_top, sims.numel())
                    if k > 0:
                        vals, flat_idx = torch.topk(sims.view(-1), k, largest=True, sorted=True)
                        rows = (flat_idx // Pb).to(torch.int64)
                        cols = (flat_idx %  Pb).to(torch.int64)

                        vals = vals.detach().cpu().tolist()
                        rows = rows.detach().cpu().tolist()
                        cols = cols.detach().cpu().tolist()

                        for s, r, c in zip(vals, rows, cols):
                            ia, jb = ai + r, bj + c
                            if len(heap) < keep_cap:
                                push(heap, (s, ia, jb))
                            elif s > heap[0][0]:
                                pushpop(heap, (s, ia, jb))

                    # cleanup per tile
                    del sims
                    if device == "cuda": torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and (Ablk > 512 or Bblk > 512):
                Ablk = max(512, Ablk // 2); Bblk = max(512, Bblk // 2)
                log(f"[OOM] Reducing tile sizes to A_BLOCK={Ablk}, B_BLOCK={Bblk} and retrying…")
                if device == "cuda": torch.cuda.empty_cache()
                heap.clear()
                continue
            raise

    heap.sort(key=lambda x: x[0], reverse=True)
    return [(ia, jb, float(s)) for (s, ia, jb) in heap]

def nms_temporal_frameidx(matches, min_sep_a: int, min_sep_b: int):
    """Keep matches if they are separated by >= min_sep_* on BOTH A and B timelines."""
    kept = []
    for (i, j, s) in matches:
        ok = True
        for (_i, _j, _s) in kept:
            if abs(i - _i) < min_sep_a or abs(j - _j) < min_sep_b:
                ok = False; break
        if ok: kept.append((i, j, s))
    return kept

def time_from_index(video_path: str, frame_index: int, stride: int) -> float:
    fps = get_fps(video_path)
    return (frame_index * stride) / max(fps, 1e-6)

def save_frame_at_time(video_path: str, t_sec: float, out_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t_sec) * 1000.0)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None: return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return cv2.imwrite(out_path, frame_bgr)

def main():
    p = argparse.ArgumentParser(description="Find similar frames between two videos fast (GPU, OOM-safe).")
    p.add_argument("video_a")
    p.add_argument("video_b")
    p.add_argument("--outdir", default=os.path.expanduser("~/Videos/matched_frames"))
    p.add_argument("--topk", type=int, default=1, help="How many best non-overlapping pairs to save.")
    p.add_argument("--min-sep-sec", type=float, default=2.0, help="Minimum separation between picks (seconds).")
    p.add_argument("--stride", type=int, default=1, help="Sample every Nth frame for speed (1=native).")
    p.add_argument("--model", default="dinov2", choices=["dinov2","resnet50"])
    p.add_argument("--device", default="auto", help="auto/cuda/cpu")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # (Optional) allocator hints for fewer OOMs due to fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    log(f"Loading model '{args.model}' on '{args.device}'…")
    model, input_hw, mean, std, device = load_feat_extractor(args.model, args.device)
    log(f"Model ready on {device} (AMP on).")

    # Embed A
    fps_a = get_fps(args.video_a)
    approx_a = fps_a / max(1, args.stride)
    init_batch_a = 12 if args.model == "dinov2" else 64
    log(f"Sampling {args.video_a} every {args.stride} → ~{approx_a:.2f} fps")
    feats_a = embed_video(args.video_a, model, input_hw, mean, std, device,
                          stride=args.stride, init_batch=init_batch_a)
    log(f"A: frames={feats_a.shape[0]}, dim={feats_a.shape[1]}")
    if feats_a.shape[0] == 0:
        log("No frames sampled from A. Aborting."); sys.exit(1)
    if device == "cuda": torch.cuda.empty_cache()

    # Embed B
    fps_b = get_fps(args.video_b)
    approx_b = fps_b / max(1, args.stride)
    init_batch_b = 12 if args.model == "dinov2" else 64
    log(f"Sampling {args.video_b} every {args.stride} → ~{approx_b:.2f} fps")
    feats_b = embed_video(args.video_b, model, input_hw, mean, std, device,
                          stride=args.stride, init_batch=init_batch_b)
    log(f"B: frames={feats_b.shape[0]}, dim={feats_b.shape[1]}")
    if feats_b.shape[0] == 0:
        log("No frames sampled from B. Aborting."); sys.exit(1)
    if device == "cuda": torch.cuda.empty_cache()

    # GPU similarity (tiling) with OOM backoff
    log("Scanning similarities on GPU (tiled)…")
    candidates = top_candidates_gpu(
        feats_a, feats_b, device,
        keep_cap=max(GLOBAL_HEAP_CAP, args.topk * 8000),
        subblock_top=SUBBLOCK_TOP,
        a_block_init=A_BLOCK_INIT, b_block_init=B_BLOCK_INIT
    )
    if not candidates:
        log("No similarity candidates found."); sys.exit(2)

    # Temporal NMS using per-side separations (in sampled-frame units)
    min_sep_a = int(round(args.min_sep_sec * approx_a))
    min_sep_b = int(round(args.min_sep_sec * approx_b))
    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = nms_temporal_frameidx(candidates, min_sep_a, min_sep_b)[:args.topk]
    if not selected:
        log("No matches after temporal filtering. Try lowering --min-sep-sec."); sys.exit(3)

    # Save frames
    for rank, (ia, ib, s) in enumerate(selected, 1):
        ta = time_from_index(args.video_a, ia, args.stride)
        tb = time_from_index(args.video_b, ib, args.stride)
        a_path = os.path.join(args.outdir, f"{rank:04d}_A_{ta:09.3f}s.png")
        b_path = os.path.join(args.outdir, f"{rank:04d}_B_{tb:09.3f}s.png")
        ok_a = save_frame_at_time(args.video_a, ta, a_path)
        ok_b = save_frame_at_time(args.video_b, tb, b_path)
        status = "OK" if (ok_a and ok_b) else f"FAIL(A={ok_a},B={ok_b})"
        log(f"[{rank}] cos={s:.6f} | A@{ta:.3f}s -> {a_path} | B@{tb:.3f}s -> {b_path} [{status}]")

if __name__ == "__main__":
    main()

