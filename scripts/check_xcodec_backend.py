import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Check optional Transformers X-Codec backend")
    parser.add_argument("--model-path", default=None, help="Local path or HF id, e.g. hf-audio/xcodec-wavlm-more-data")
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import transformers
        from transformers import XcodecModel
    except ImportError as exc:
        print(f"ERROR Transformers XcodecModel is unavailable: {exc}")
        return 1
    print(f"transformers {transformers.__version__} {transformers.__file__}")
    print("XcodecModel API is available")
    if not args.model_path:
        print("No --model-path provided; skipped weight loading and encode check")
        return 0

    from model.hybrid_xcodec import XCodecFirstRVQTokenizer

    tokenizer = XCodecFirstRVQTokenizer(
        vocab_size=args.vocab_size,
        backend="model.xcodec_backends:TransformersXCodecFirstRVQ",
        model_path=args.model_path,
        rvq_axis=1,
        backend_kwargs={
            "token_attr": "audio_codes",
            "device": args.device,
        },
    )
    wav = torch.zeros(1, int(args.samples), dtype=torch.float32)
    batch = tokenizer.encode_first_rvq_batch(wav)
    print("tokens", tuple(batch.tokens.shape), batch.tokens.dtype)
    print("mask", tuple(batch.mask.shape), int(batch.mask.sum().item()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
