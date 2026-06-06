from __future__ import annotations

from typing import Any
import inspect

import torch


class TransformersXCodecFirstRVQ:
    """Optional Hugging Face/Transformers X-Codec tokenizer backend.

    Official Transformers X-Codec models expose `XcodecModel.encode`, whose
    output includes `audio_codes` shaped `[B, Q, T]`. This adapter also keeps a
    generic extraction fallback for custom X-Codec releases.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        processor_path: str | None = None,
        encode_method: str = "encode",
        token_attr: str | None = None,
        trust_remote_code: bool = True,
        device: str | None = None,
        bandwidth: float | None = None,
    ):
        if model_path is None:
            raise ValueError("TransformersXCodecFirstRVQ requires xcodec.model_path")
        try:
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "transformers is required for TransformersXCodecFirstRVQ"
            ) from exc

        self.model_path = model_path
        self.processor_path = processor_path or model_path
        self.encode_method = encode_method
        self.token_attr = token_attr
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.bandwidth = bandwidth
        try:
            from transformers import XcodecModel

            model_cls = XcodecModel
        except ImportError:
            model_cls = AutoModel
        self.model = model_cls.from_pretrained(model_path, trust_remote_code=trust_remote_code).to(self.device)
        self.model.eval()
        try:
            from transformers import AutoFeatureExtractor

            self.processor = AutoFeatureExtractor.from_pretrained(
                self.processor_path,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            try:
                self.processor = AutoProcessor.from_pretrained(self.processor_path, trust_remote_code=trust_remote_code)
            except Exception:
                self.processor = None
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def __call__(self, clean_wav_16k: torch.Tensor, sample_rate: int = 16000):
        wav = clean_wav_16k.to(self.device)
        inputs: dict[str, Any] | torch.Tensor
        if self.processor is not None:
            examples = [item for item in wav.detach().cpu().numpy()]
            processed = self.processor(
                examples,
                sampling_rate=int(sample_rate),
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in processed.items()
            }
        else:
            inputs = wav

        with torch.no_grad():
            method = getattr(self.model, self.encode_method)
            kwargs: dict[str, Any] = {}
            if self.bandwidth is not None:
                kwargs["bandwidth"] = float(self.bandwidth)
            if isinstance(inputs, dict):
                signature = inspect.signature(method)
                allowed = set(signature.parameters)
                call_inputs = {key: value for key, value in inputs.items() if key in allowed}
                output = method(**call_inputs, **kwargs)
            else:
                output = method(inputs, **kwargs)
        return self._extract_tokens(output, clean_wav_16k.device)

    def _extract_tokens(self, output: Any, return_device: torch.device):
        if self.token_attr is not None:
            tokens = self._lookup_attr(output, self.token_attr)
            return {"tokens": tokens.to(return_device)}
        if torch.is_tensor(output):
            return {"tokens": output.to(return_device)}
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item) and item.ndim in {2, 3}:
                    return {"tokens": item.to(return_device)}
        if isinstance(output, dict):
            for key in ("codes", "tokens", "token_ids", "indices", "codebook_indices", "audio_codes"):
                value = output.get(key)
                if torch.is_tensor(value):
                    result = {"tokens": value.to(return_device)}
                    mask = output.get("mask")
                    if mask is None:
                        mask = output.get("attention_mask")
                    if torch.is_tensor(mask):
                        result["mask"] = mask.to(return_device)
                    return result
        for key in ("codes", "tokens", "token_ids", "indices", "codebook_indices", "audio_codes"):
            if hasattr(output, key):
                value = getattr(output, key)
                if torch.is_tensor(value):
                    return {"tokens": value.to(return_device)}
        raise ValueError(
            "Could not find X-Codec tokens in backend output. Set "
            "xcodec.backend_kwargs.token_attr to the output field containing RVQ IDs."
        )

    @staticmethod
    def _lookup_attr(output: Any, path: str):
        value = output
        for part in path.split("."):
            if isinstance(value, dict):
                value = value[part]
            else:
                value = getattr(value, part)
        if not torch.is_tensor(value):
            raise TypeError(f"Configured X-Codec token_attr {path!r} did not resolve to a tensor")
        return value
