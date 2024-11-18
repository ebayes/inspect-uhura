# run.py

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import scorer, accuracy, bootstrap_std
from inspect_ai.solver import multiple_choice, TaskState, system_message
from inspect_ai.scorer._target import Target
from inspect_ai.scorer._metric import Score
import os

# Import the prompts
from custom_scripts.prompts import PROMPT_1, PROMPT_2, PROMPT_3, PROMPT_4, PROMPT_5

# Read the few_shot value from the environment variable, defaulting to 5
few_shot = int(os.environ.get("FEW_SHOT", "5"))

# Read the PROMPT_ID from the environment variable, defaulting to 1
prompt_id = int(os.environ.get("PROMPT_ID", "1"))

# Map PROMPT_ID to the corresponding prompt
prompt_dict = {1: PROMPT_1, 2: PROMPT_2, 3: PROMPT_3, 4: PROMPT_4, 5: PROMPT_5}

# Get the selected prompt; default to PROMPT_1 if invalid ID
selected_prompt = prompt_dict.get(prompt_id, PROMPT_1)


def sample_to_fewshot(sample):
    choices_text = "\n".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample.choices)]
    )
    return f"Question: {sample.input}\n\nOptions:\n{choices_text}\n\nCorrect Answer: {sample.target[0]}"


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
            return Score(
                value="C" if is_correct else "I",
                answer=model_answer,
                explanation=f"Model answer: {model_answer}, Correct answers: {correct_answers}, No logprobs available",
            )

        avg_logprob = sum(lp.logprob for lp in logprobs.content) / len(logprobs.content)

        return Score(
            value="C" if is_correct else "I",
            answer=model_answer,
            explanation=(
                f"Model answer: {model_answer}, Correct answers: {correct_answers}, "
                f"Average log probability: {avg_logprob:.4f}"
            ),
            metadata={"avg_logprob": avg_logprob},
        )

    return score


def labels_to_positions(labels: list[int]) -> list[str]:
    return [chr(ord("A") + i) for i, label in enumerate(labels) if label == 1]


def truthfulqa_task(lang_code, target="mc1"):
    def record_to_sample(record):
        return Sample(
            input=record["question"],
            choices=record[f"{target}_targets"]["choices"],
            target=labels_to_positions(record[f"{target}_targets"]["labels"]),
        )

    # Get few_shot examples from the training split
    if few_shot > 0:
        fewshots = hf_dataset(
            path="ebayes/uhura-truthfulqa-clean",
            name=f"{lang_code}_multiple_choice",
            sample_fields=record_to_sample,
            split="train",
            shuffle=True,
            seed=42,
            limit=few_shot,
        )
        fewshot_examples = "\n\n".join(
            [sample_to_fewshot(sample) for sample in fewshots]
        )
        # Prepare the system message
        system_msg = (
            f"{selected_prompt}\n\n"
            f"Here are some example questions and answers:\n\n{fewshot_examples}"
        )
    else:
        system_msg = f"{selected_prompt}"

    dataset = hf_dataset(
        path="ebayes/uhura-truthfulqa-clean",
        name=f"{lang_code}_multiple_choice",
        sample_fields=record_to_sample,
        split="test",
        shuffle=True,
    )

    multiple_correct = target != "mc1"

    return Task(
        dataset=dataset,
        plan=[
            system_message(system_msg),
            multiple_choice(
                multiple_correct=multiple_correct,
                shuffle=True,
                # Do not pass a custom template here
            ),
        ],
        scorer=logprob_based_scorer(),
    )


@task
def amharic():
    return truthfulqa_task("am")


@task
def hausa():
    return truthfulqa_task("ha")


@task
def sotho():
    return truthfulqa_task("nso")


@task
def swahili():
    return truthfulqa_task("sw")


@task
def yoruba():
    return truthfulqa_task("yo")


@task
def english():
    return truthfulqa_task("en")


@task
def zulu():
    return truthfulqa_task("zu")
