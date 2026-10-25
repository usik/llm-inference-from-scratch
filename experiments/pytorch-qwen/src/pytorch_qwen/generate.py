from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"
DEFAULT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


@dataclass(frozen=True)
class Result:
    model_id: str
    revision: str
    device: str
    dtype: str
    prompt_tokens: int
    output_tokens: int
    model_load_seconds: float
    generation_seconds: float
    output_tokens_per_second: float
    mps_allocated_bytes: int
    mps_driver_allocated_bytes: int
    python_version: str
    torch_version: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Qwen generation on PyTorch MPS.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--prompt",
        default="LLM 추론에서 KV cache가 필요한 이유를 두 문장으로 설명해줘.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--thinking", action="store_true")
    return parser.parse_args()


def require_mps() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is not available. Run qwen-preflight first.")


def main() -> None:
    args = parse_args()
    require_mps()
    torch.manual_seed(args.seed)

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("mps")
    model.eval()
    torch.mps.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to("mps")
    prompt_tokens = inputs["input_ids"].shape[-1]

    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.6 if args.thinking else 0.7,
            top_p=0.95 if args.thinking else 0.8,
            top_k=20,
        )
    torch.mps.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    output_ids = generated[0, prompt_tokens:]
    output_tokens = output_ids.shape[-1]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    result = Result(
        model_id=args.model_id,
        revision=args.revision,
        device="mps",
        dtype="float16",
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        model_load_seconds=model_load_seconds,
        generation_seconds=generation_seconds,
        output_tokens_per_second=(
            output_tokens / generation_seconds if generation_seconds else 0.0
        ),
        mps_allocated_bytes=torch.mps.current_allocated_memory(),
        mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
    )

    payload = {"result": asdict(result), "prompt": args.prompt, "output": output_text}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
