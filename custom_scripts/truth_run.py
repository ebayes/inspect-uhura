import argparse
import pandas as pd
from inspect_ai import Task, task, ModelSpec
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import scorer, accuracy, bootstrap_std
from inspect_ai.solver import multiple_choice, system_message, TaskState
from inspect_ai.scorer._target import Target
from inspect_ai.scorer._metric import Score

lang_code = "yo"  # Change this to the language code you want to test
few_shot = 5  # You can change this to any number you want


def sample_to_fewshot(sample):
    choices_text = "\n".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample.choices)]
    )
    return f"Question: {sample.input}\n\nChoices:\n{choices_text}\n\nAnswer: {sample.target[0]}"


@scorer(metrics=[accuracy(), bootstrap_std()])
def logprob_based_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        model_answer_full = state.output.completion.strip().upper()
        # Extract just the letter from the model's answer
        model_answer = model_answer_full.split(":")[-1].strip()

        # Handle the case where target.target is a list
        if isinstance(target.target, list):
            correct_answers = [ans.strip().upper() for ans in target.target]
        else:
            correct_answers = [str(target.target).strip().upper()]

        is_correct = model_answer in correct_answers

        logprobs = state.output.choices[0].logprobs

        if logprobs is None:
            avg_logprob = None
        else:
            avg_logprob = sum(lp.logprob for lp in logprobs.content) / len(
                logprobs.content
            )

        return Score(
            value="C" if is_correct else "I",
            answer=model_answer,
            explanation=f"Model answer: {model_answer}, Correct answers: {correct_answers}, Average log probability: {avg_logprob}",
            metadata={"avg_logprob": avg_logprob},
        )

    return score


def labels_to_positions(labels):
    return [chr(ord("A") + i) for i, label in enumerate(labels) if label == 1]


@task
def truthfulqa(target="mc1"):
    def record_to_sample(record):
        return Sample(
            input=record["question"],
            choices=record[f"{target}_targets"]["choices"],
            target=labels_to_positions(record[f"{target}_targets"]["labels"]),
        )

    # Get few-shot examples from the training split
    fewshots = hf_dataset(
        path="ebayes/uhura-eval",
        name=f"{lang_code}_multiple_choice",
        sample_fields=record_to_sample,
        split="train",
        shuffle=True,
        seed=42,
        limit=few_shot,
    )

    fewshot_examples = "\n\n".join([sample_to_fewshot(sample) for sample in fewshots])

    # Load the test dataset
    dataset = hf_dataset(
        path="ebayes/uhura-eval",
        name=f"{lang_code}_multiple_choice",
        sample_fields=record_to_sample,
        split="test",
        shuffle=True,
    )

    if target == "mc1":
        multiple_correct = False
    else:
        multiple_correct = True

    return Task(
        dataset=dataset,
        plan=[
            system_message(
                f"Here are some example questions and answers:\n\n{fewshot_examples}"
            ),
            multiple_choice(multiple_correct=multiple_correct, shuffle=True),
        ],
        scorer=logprob_based_scorer(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TruthfulQA evaluation.")
    parser.add_argument(
        "--model", type=str, default="openai/gpt-3.5-turbo", help="Model to evaluate."
    )
    parser.add_argument(
        "--output", type=str, default="results.csv", help="Output CSV file."
    )
    args = parser.parse_args()

    # Get the task
    task_instance = truthfulqa()

    # Define the model specification
    if "/" in args.model:
        provider, model_name = args.model.split("/", 1)
    else:
        provider, model_name = "openai", args.model  # Default to 'openai' provider

    model_spec = ModelSpec(
        provider=provider,
        model=model_name,
    )

    # Run the evaluation
    results = task_instance.evaluate(model=model_spec)

    # Convert results to pandas DataFrame
    results_df = results.to_pandas()

    # Save to CSV
    results_df.to_csv(args.output, index=False)

    print(f"Results saved to {args.output}")
