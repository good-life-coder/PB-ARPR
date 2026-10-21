"""Shared exact AU-PRO from validated Phase 07C-R1."""
import gc,math,shutil,time
from pathlib import Path
ALPHAS=(0.05,0.30)

def pro_at_fprs(background, scores, weights, fprs, np):
    """Pointwise PRO of the same tie-aware curve integrated by integrate.

    At a vertical jump, return the right-continuous (upper) PRO value.
    This does not select an operating threshold or alter the integration formula.
    """
    n = len(background)
    if n == 0:
        raise ValueError("Background pixels required")
    values = np.zeros(len(fprs), dtype=np.float64)
    for start in range(0, len(scores), 250_000):
        s = scores[start:start+250_000]
        w = np.asarray(weights[start:start+250_000], dtype=np.float64)
        left = np.searchsorted(background, s, side="left")
        right = np.searchsorted(background, s, side="right")
        q_gt = (n-right).astype(np.float64)/n
        q_eq = (right-left).astype(np.float64)/n
        ties = right > left
        for index, alpha in enumerate(fprs):
            contribution = (alpha >= q_gt).astype(np.float64)
            contribution[ties] = np.clip((alpha-q_gt[ties])/q_eq[ties], 0, 1)
            values[index] += float(np.dot(w, contribution))
    return values

def integrate(background, scores, weights, np):
    """All rank comparisons use unchanged float32 scores; arithmetic is float64."""
    n = len(background)
    if not n or not len(scores):
        raise ValueError("Both background pixels and positive regions are required")
    totals = {alpha: 0.0 for alpha in ALPHAS}
    weight_sum = 0.0
    for start in range(0, len(scores), 250_000):
        s = scores[start:start + 250_000]
        w = np.asarray(weights[start:start + 250_000], dtype=np.float64)
        left = np.searchsorted(background, s, side="left")
        right = np.searchsorted(background, s, side="right")
        q_gt = (n - right).astype(np.float64) / n
        q_ge = (n - left).astype(np.float64) / n
        q_eq = (right - left).astype(np.float64) / n
        weight_sum += float(w.sum(dtype=np.float64))
        for alpha in ALPHAS:
            area = np.zeros(len(s), dtype=np.float64)
            active = alpha > q_gt
            no_tie = active & (right == left)
            full_tie = active & (right > left) & (alpha >= q_ge)
            partial_tie = active & (right > left) & (alpha < q_ge)
            area[no_tie] = alpha - q_gt[no_tie]
            area[full_tie] = q_eq[full_tie] / 2 + alpha - q_ge[full_tie]
            area[partial_tie] = (alpha - q_gt[partial_tie]) ** 2 / (2 * q_eq[partial_tie])
            totals[alpha] += float(np.dot(w, area))
    return {alpha: value / alpha for alpha, value in totals.items()}, weight_sum

def components(mask, cv2, np):
    # cv2.connectedComponents default is 8, matching installed anomalib helper.
    count, labels = cv2.connectedComponents(np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8)
    sizes = np.bincount(labels.reshape(-1), minlength=count)
    return labels, sizes

def get_mask(row, Image, np):
    if int(row["gt_label"]) == 0:
        return np.zeros(row["native_shape"], dtype=np.uint8)
    with Image.open(row["gt_mask_path"]) as image:
        return (np.asarray(image.convert("L")) > 0).astype(np.uint8)

def get_score(row, torch, np):
    score = np.load(row["raw_map_path"], allow_pickle=False)
    if score.shape != row["native_shape"]:
        score = torch.nn.functional.interpolate(torch.from_numpy(score)[None, None], size=row["native_shape"],
                                               mode="bilinear", align_corners=False)[0, 0].numpy()
    return score

def native(rows, report, torch, cv2, np, Image, *, temp_dir):
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    nb = npos = k = 0
    regions = []
    for row in rows:
        labels, sizes = components(get_mask(row, Image, np), cv2, np)
        nb += int(sizes[0])
        npos += int(sizes[1:].sum())
        k += len(sizes) - 1
        regions.append(dict(sample_id=row["sample_id"], region_sizes=sizes[1:].tolist()))
    if not nb or not npos or not k:
        raise ValueError("Native GT lacks background or connected regions")
    report.update(background_pixels=nb, positive_pixels=npos, connected_regions=k, regions=regions)
    if shutil.disk_usage(temp_dir).free < nb * 4 + npos * 12 + 512 * 1024 ** 2:
        raise OSError("Insufficient disk space for exact-sized memmaps")
    bg = np.memmap(temp_dir / "background_scores.dat", mode="w+", dtype=np.float32, shape=(nb,))
    pos = np.memmap(temp_dir / "positive_scores.dat", mode="w+", dtype=np.float32, shape=(npos,))
    weights = np.memmap(temp_dir / "positive_weights.dat", mode="w+", dtype=np.float64, shape=(npos,))
    bi = pi = 0
    for index, row in enumerate(rows):
        score = get_score(row, torch, np)
        labels, sizes = components(get_mask(row, Image, np), cv2, np)
        fg = labels > 0
        n = int(fg.sum())
        b = fg.size - n
        bg[bi:bi+b] = score[~fg]
        pos[pi:pi+n] = score[fg]
        weights[pi:pi+n] = 1.0 / (k * sizes[labels[fg]].astype(np.float64))
        bi += b
        pi += n
        if (index + 1) % 10 == 0:
            print(f"Native memmap write: {index+1}/114", flush=True)
    if bi != nb or pi != npos:
        raise ValueError("Native pass counts changed")
    bg.flush()
    pos.flush()
    weights.flush()
    total = sum(float(weights[i:i+1_000_000].sum(dtype=np.float64)) for i in range(0, npos, 1_000_000))
    report["weight_sum"] = total
    if abs(total - 1) >= 1e-6:
        raise ArithmeticError(f"Region weights sum to {total}")
    print(f"Sorting {nb:,} background scores in-place on CPU...", flush=True)
    sort_start = time.perf_counter()
    bg.sort(kind="quicksort")
    bg.flush()
    report["sorting_elapsed_seconds"] = time.perf_counter() - sort_start
    # Full chunked monotonicity check, including all chunk boundaries.
    for i in range(0, nb-1, 1_000_000):
        stop = min(nb-1, i+1_000_000)
        if np.any(bg[i:stop] > bg[i+1:stop+1]):
            raise ValueError("Background sort monotonicity failed")
    report.update(background_min=float(bg[0]), background_max=float(bg[-1]), sort_verified="full chunked monotonicity")
    results, weight_sum = integrate(bg, pos, weights, np)
    if abs(weight_sum - 1) >= 1e-6:
        raise ArithmeticError(f"Integration weights sum to {weight_sum}")
    for alpha, value in results.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Native AU-PRO outside [0,1]: {value}")
        report[f"aupro_{alpha:.2f}"] = value
        report[f"aupro_{alpha:.2f}_percent"] = 100 * value
    del bg, pos, weights
    gc.collect()
