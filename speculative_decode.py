import argparse
import os
import time
from typing import Tuple

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


DEFAULT_DRAFT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_TARGET_MODEL = "mlx-community/Qwen3-8B-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal speculative decoding demo with MLX."
    )
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT_MODEL,
        help="Path to the smaller draft model.",
    )
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help="Path to the larger target model.",
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt to run. If omitted, start an interactive chat loop.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt used when the tokenizer has a chat template.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="How many draft tokens to propose per speculative step.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold.",
    )
    return parser.parse_args()


def ensure_model_ref(path_or_repo: str, label: str) -> None:
    is_local_dir = os.path.isdir(path_or_repo)
    looks_like_repo_id = "/" in path_or_repo and not path_or_repo.startswith("/")
    if not is_local_dir and not looks_like_repo_id:
        raise FileNotFoundError(
            f"{label} model not found as a local directory or repo id: {path_or_repo}"
        )


def format_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return user_prompt


def load_models(target_path: str, draft_path: str):
    print(f"Loading target model: {target_path}")
    target_model, tokenizer = load(target_path)
    print(f"Loading draft model:  {draft_path}")
    draft_model, draft_tokenizer = load(draft_path)

    if getattr(tokenizer, "vocab_size", None) != getattr(
        draft_tokenizer, "vocab_size", None
    ):
        print(
            "Warning: tokenizer vocab sizes differ. "
            "Speculative decoding requires compatible tokenization."
        )

    return target_model, tokenizer, draft_model


def generate_once(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    *,
    system_prompt: str,
    max_tokens: int,
    num_draft_tokens: int,
    temp: float,
    top_p: float,
) -> Tuple[str, int, int, float, float, float, float]:
    formatted_prompt = format_prompt(tokenizer, system_prompt, prompt)
    sampler = make_sampler(temp=temp, top_p=top_p)

    pieces = []
    accepted_from_draft = 0
    generated_tokens = 0
    prompt_tps = 0.0
    generation_tps = 0.0
    peak_memory = 0.0
    started_at = time.perf_counter()

    for response in stream_generate(
        target_model,
        tokenizer,
        formatted_prompt,
        max_tokens=max_tokens,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        sampler=sampler,
    ):
        if response.text:
            print(response.text, end="", flush=True)
            pieces.append(response.text)
        if response.from_draft:
            accepted_from_draft += 1
        generated_tokens = response.generation_tokens
        prompt_tps = response.prompt_tps
        generation_tps = response.generation_tps
        peak_memory = response.peak_memory

    elapsed = time.perf_counter() - started_at
    print()
    return (
        "".join(pieces),
        accepted_from_draft,
        generated_tokens,
        prompt_tps,
        generation_tps,
        peak_memory,
        elapsed,
    )


def print_stats(
    accepted_from_draft: int,
    generated_tokens: int,
    prompt_tps: float,
    generation_tps: float,
    peak_memory: float,
    elapsed: float,
) -> None:
    acceptance_rate = (
        accepted_from_draft / generated_tokens if generated_tokens > 0 else 0.0
    )
    print(f"accepted_from_draft: {accepted_from_draft}/{generated_tokens}")
    print(f"acceptance_rate:     {acceptance_rate:.2%}")
    print(f"prompt_tps:          {prompt_tps:.2f}")
    print(f"generation_tps:      {generation_tps:.2f}")
    print(f"peak_memory_gb:      {peak_memory:.2f}")
    print(f"wall_time_sec:       {elapsed:.2f}")


def main() -> None:
    args = parse_args()
    ensure_model_ref(args.target_model, "Target")
    ensure_model_ref(args.draft_model, "Draft")

    target_model, tokenizer, draft_model = load_models(
        args.target_model,
        args.draft_model,
    )

    if args.prompt:
        print("Assistant: ", end="", flush=True)
        _, accepted, generated, prompt_tps, generation_tps, peak_memory, elapsed = (
            generate_once(
                target_model,
                draft_model,
                tokenizer,
                args.prompt,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                num_draft_tokens=args.num_draft_tokens,
                temp=args.temp,
                top_p=args.top_p,
            )
        )
        print_stats(
            accepted,
            generated,
            prompt_tps,
            generation_tps,
            peak_memory,
            elapsed,
        )
        return

    print("Interactive speculative decoding chat. Type 'exit' to quit.")
    while True:
        user_prompt = input("\nYou: ").strip()
        if not user_prompt or user_prompt.lower() in {"exit", "quit"}:
            break

        print("Assistant: ", end="", flush=True)
        _, accepted, generated, prompt_tps, generation_tps, peak_memory, elapsed = (
            generate_once(
                target_model,
                draft_model,
                tokenizer,
                user_prompt,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                num_draft_tokens=args.num_draft_tokens,
                temp=args.temp,
                top_p=args.top_p,
            )
        )
        print_stats(
            accepted,
            generated,
            prompt_tps,
            generation_tps,
            peak_memory,
            elapsed,
        )


if __name__ == "__main__":
    main()
