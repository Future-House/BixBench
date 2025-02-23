from datasets import load_dataset
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))
import ast
from bixbench import ZeroshotBaseline, AgentInput
import argparse
import sys
from itertools import islice
import asyncio

hf_url = "futurehouse/BixBench-internal"


def parse_args():
    parser = argparse.ArgumentParser(
        description="BixBench Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--eval-mode",
        choices=["mcq", "openanswer"],
        default="mcq",
        help="Evaluation mode",
    )
    parser.add_argument("--model", default="gpt-4o", help="Model name to use")
    
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Model temperature"
    )

    parser.add_argument(
        "--with-refusal",
        action="store_true",
        default=False,
        help="Add refusal option for MCQs",
    )

    # used for testing purposes
    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="Number of examples to evaluate. Default is -1 for all examples",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results. Default is 'results'",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="Output file name (optional)"
    )
    return parser.parse_args()


def string_to_list(example):
    example["questions"] = ast.literal_eval(example["questions"])
    example["categories"] = ast.literal_eval(example["categories"])
    return example


async def evaluate(
    baseline_agent: ZeroshotBaseline,
    num_examples=-1,
    output_dir="results",
    output_file=None,
) -> pd.DataFrame:
    # Load dataset
    dataset = load_dataset(hf_url)
    # map string to list
    dataset = dataset.map(string_to_list)
    results = []
    num_examples = num_examples if num_examples > 0 else len(dataset["train"])
    for i, row in enumerate(islice(dataset["train"], num_examples)):
        for q_dict in row["questions"]:
            input = AgentInput(
                id=dataset["train"][i]["uuid"],
                question=q_dict["question"],
                target=q_dict["ideal_answer"],
                choices=[q_dict[f"distractor_{i}"] for i in range(1, 4)],
            )
            answer, target, unsure_answer = (
                await baseline_agent.generate_zeroshot_answers(input)
            )
            results.append(
                {
                    "uuid": dataset["train"][i]["uuid"],
                    "question": input.question,
                    "predicted": answer,
                    "target": target,
                    "unsure": unsure_answer,
                }
            )

    # make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_file)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


async def main():
    try:
        print("Starting evaluation...")
        args = parse_args()
        output_file = (
            args.output_file
            if args.output_file is not None
            else f"results_{args.eval_mode}_{args.with_refusal}_{args.model}_{args.temperature}.csv"
        )
        baseline_agent = ZeroshotBaseline(
            eval_mode=args.eval_mode,
            with_refusal=args.with_refusal,
            model_name=args.model,
            temperature=args.temperature,
        )
        results = await evaluate(
            baseline_agent, args.num_examples, args.output_dir, output_file
        )
        print(f"Evaluation completed and results saved to {args.output_dir} folder")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
