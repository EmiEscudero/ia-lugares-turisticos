"""Show destination candidates and retain uncertain predictions for review."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from .inference import predict_candidates
from .learning import load_checkpoint, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.top_k < 1 or not 0 <= args.min_confidence <= 1:
        parser.error("top-k must be positive; min-confidence must be between 0 and 1.")
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    candidates = predict_candidates(model, checkpoint["classes"], image, device, args.top_k)
    print(json.dumps({"status": "uncertain" if candidates[0]["score"] < args.min_confidence else "candidate",
                      "candidates": candidates,
                      "note": "Softmax scores are not calibrated probabilities of correctness or an unknown-place detector."}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
