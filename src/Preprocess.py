import os
import argparse
import numpy as np
import cv2


# Fundus masking
def mask_fundus_circle(img: np.ndarray, threshold: int = 10)->tuple:
    """Mask fundus region of an image by detecting the largest circular contour."""
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img, None
    largest = max(contours, key=cv2.contourArea)
    mask_filled = np.zeros_like(gray)
    cv2.drawContours(mask_filled, [largest], -1, 255, thickness=cv2.FILLED)
    masked = cv2.bitwise_and(img, img, mask=mask_filled)
    return masked, mask_filled


# Artifact detection
def is_crescent_artifact(
    img: np.ndarray,
    mask,
    outer_ring_width: int = 30,
    flare_ratio_thresh: float = 1.4,
    min_flare_area_ratio: float = 0.02,
)->bool:
    """Detect bright crescent or flare artifacts near the fundus edge."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    h, w = gray.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_ring_width * 2, outer_ring_width * 2))
    eroded = cv2.erode(mask, kernel)
    ring_mask = cv2.subtract(mask, eroded)

    fundus_area = np.sum(mask == 255)
    ring_area = np.sum((gray > 200) & (ring_mask == 255))
    area_ratio = ring_area / max(fundus_area, 1)
    if area_ratio < min_flare_area_ratio:
        return False

    center_mean = np.mean(gray[eroded == 255])
    ring_mean = np.mean(gray[ring_mask == 255])

    edge_pct = 0.2
    edge_masks = {
        "top":    np.zeros_like(mask),
        "bottom": np.zeros_like(mask),
        "left":   np.zeros_like(mask),
        "right":  np.zeros_like(mask),
    }
    edge_masks["top"][:int(h * edge_pct), :] = mask[:int(h * edge_pct), :]
    edge_masks["bottom"][int(h * (1 - edge_pct)):, :] = mask[int(h * (1 - edge_pct)):, :]
    edge_masks["left"][:, :int(w * edge_pct)] = mask[:, :int(w * edge_pct)]
    edge_masks["right"][:, int(w * (1 - edge_pct)):] = mask[:, int(w * (1 - edge_pct)):]

    edge_brightness = {d: np.mean(gray[m == 255]) for d, m in edge_masks.items()}
    any_edge_spike = any(val > flare_ratio_thresh * center_mean for val in edge_brightness.values())
    ring_spike = ring_mean > flare_ratio_thresh * center_mean

    return ring_spike or any_edge_spike


def _ensure_uint8(img: np.ndarray)->np.ndarray:
    """Convert image to uint8 format in case it isn't."""
    if img.dtype != np.uint8:
        return (img * 255).astype(np.uint8)
    return img


def clean_and_collect(images: np.ndarray, labels: np.ndarray, bad_dir: str,
                      outer_ring_width=30, flare_ratio_thresh=1.4, min_flare_area_ratio=0.02)->tuple:
    """Filter images by removing artifact-contaminated ones and save rejected samples."""
    os.makedirs(bad_dir, exist_ok=True)
    clean_images, clean_labels = [], []

    for idx in range(len(images)):
        img = _ensure_uint8(images[idx])

        masked_img, mask = mask_fundus_circle(img)
        if mask is not None and not is_crescent_artifact(
            img, mask,
            outer_ring_width=outer_ring_width,
            flare_ratio_thresh=flare_ratio_thresh,
            min_flare_area_ratio=min_flare_area_ratio
        ):
            clean_images.append(masked_img)
            clean_labels.append(labels[idx])
        else:
            label_val = labels[idx][0] if labels.ndim == 2 else labels[idx]
            filename = f"{idx}_{int(label_val)}.png"
            save_path = os.path.join(bad_dir, filename)
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if len(clean_images) == 0:
        return np.empty((0, *images.shape[1:])), np.empty((0, *labels.shape[1:])) if labels.ndim > 1 else np.empty((0,), dtype=labels.dtype)

    return np.stack(clean_images), np.stack(clean_labels)


def shuffle_and_split(images: np.ndarray, labels: np.ndarray, train_ratio=0.6, val_ratio=0.2, seed: int = 42)->tuple:
    """Shuffle dataset and split into train/val/test sets by ratio."""
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1
    rng = np.random.default_rng(seed)
    n = len(images)
    idx = rng.permutation(n)

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    return (
        images[train_idx], labels[train_idx],
        images[val_idx], labels[val_idx],
        images[test_idx], labels[test_idx],
    )


def process_npz(
    input_npz: str,
    output_npz: str,
    bad_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    outer_ring_width: int = 30,
    flare_ratio_thresh: float = 1.4,
    min_flare_area_ratio: float = 0.02,
)->None:
  """End-to-end preprocessing pipeline: load, clean, split, and save dataset."""
    if not os.path.isfile(input_npz):
        raise FileNotFoundError(f"Input NPZ not found: {input_npz}")

    data = np.load(input_npz)

    required = ["train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"]
    for k in required:
        if k not in data.files:
            raise KeyError(f"Missing key in NPZ: {k}")

    all_images = np.concatenate([data["train_images"], data["val_images"], data["test_images"]], axis=0)
    all_labels = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]], axis=0)

    print("Loaded:", input_npz)
    print("Images:", all_images.shape, "Labels:", all_labels.shape)

    clean_images, clean_labels = clean_and_collect(
        all_images, all_labels, bad_dir,
        outer_ring_width=outer_ring_width,
        flare_ratio_thresh=flare_ratio_thresh,
        min_flare_area_ratio=min_flare_area_ratio
    )
    print("Kept images:", clean_images.shape, "Kept labels:", clean_labels.shape)

    trX, trY, vaX, vaY, teX, teY = shuffle_and_split(clean_images, clean_labels, train_ratio=train_ratio, val_ratio=val_ratio)

    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
    np.savez_compressed(
        output_npz,
        train_images=trX, train_labels=trY,
        val_images=vaX,   val_labels=vaY,
        test_images=teX,  test_labels=teY
    )
    print("Saved cleaned dataset to:", output_npz)
    print("Bad images written to:", bad_dir)


def main():
    parser = argparse.ArgumentParser(description="Clean fundus images, remove crescent artifacts, shuffle & split to new NPZ.")
    parser.add_argument("--input", required=True, help="Path to input NPZ (e.g., retinamnist_binary.npz)")
    parser.add_argument("--output", required=True, help="Path to output NPZ (e.g., retinamnist_binary_cleaned_shuffled.npz)")
    parser.add_argument("--bad-dir", default=os.getenv("PROJECT_BAD_DIR", "./bad"), help="Directory to save filtered images")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split ratio (default 0.6)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio (default 0.2)")
    parser.add_argument("--outer-ring-width", type=int, default=30)
    parser.add_argument("--flare-ratio-thresh", type=float, default=1.4)
    parser.add_argument("--min-flare-area-ratio", type=float, default=0.02)
    args = parser.parse_args()

    process_npz(
        input_npz=args.input,
        output_npz=args.output,
        bad_dir=args.bad_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        outer_ring_width=args.outer_ring_width,
        flare_ratio_thresh=args.flare_ratio_thresh,
        min_flare_area_ratio=args.min_flare_area_ratio,
    )


if __name__ == "__main__":
    main()
