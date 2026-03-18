"""Generate LLM paraphrases of template captions."""
import argparse
from pathlib import Path

from lctscap.data.paraphrase import ParaphrasePipeline
from lctscap.utils.io import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Generate LLM paraphrases")
    parser.add_argument("--input", type=str, required=True, help="Annotated JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL with paraphrases")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="LLM model for paraphrasing",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    if args.max_samples:
        samples = samples[: args.max_samples]

    pipeline = ParaphrasePipeline(model_name=args.model, batch_size=args.batch_size)
    paraphrased = pipeline.process(samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paraphrased, args.output)
    print(f"Wrote {len(paraphrased)} paraphrased samples to {args.output}")


if __name__ == "__main__":
    main()
