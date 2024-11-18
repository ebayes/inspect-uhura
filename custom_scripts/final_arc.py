# to run this script, run inspect eval arc.py@amharic --model hf/meta-llama/Meta-Llama-3-8B-Instruct in the cli
# to change the language, change the @amharic to @english, @swahili, @hausa, @northern_sotho, @yoruba

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import answer, scorer, accuracy, bootstrap_std
from inspect_ai.solver import multiple_choice, system_message, TaskState
from inspect_ai.scorer._target import Target
from inspect_ai.scorer._metric import Score
from prompts import PROMPT_1, PROMPT_2, PROMPT_3, PROMPT_4, PROMPT_5


@scorer(metrics=[accuracy(), bootstrap_std()])
def logprob_based_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        model_answer_full = state.output.completion.strip().upper()
        # Extract just the letter from the model's answer
        model_answer = model_answer_full.split(":")[-1].strip()

        # Handle the case where target.target is a list
        if isinstance(target.target, list):
            correct_answer = target.target[0].strip().upper() if target.target else ""
        else:
            correct_answer = str(target.target).strip().upper()

        is_correct = model_answer == correct_answer

        logprobs = state.output.choices[0].logprobs

        # print(f"Logprobs available: {logprobs is not None}")

        if logprobs is None:
            return Score(
                value="C" if is_correct else "I",
                answer=model_answer,
                explanation=f"Model answer: {model_answer}, Correct answer: {correct_answer}, No logprobs available",
            )

        avg_logprob = sum(lp.logprob for lp in logprobs.content) / len(logprobs.content)

        return Score(
            value="C" if is_correct else "I",
            answer=model_answer,
            explanation=f"Model answer: {model_answer}, Correct answer: {correct_answer}, Average log probability: {avg_logprob:.4f}",
            metadata={"avg_logprob": avg_logprob},
        )

    return score


def record_to_sample(record):
    # read the choices
    choices = record["choices"]

    # Parse the JSON string if choices is a string
    if isinstance(choices, str):
        import json

        choices_dict = json.loads(choices)
    else:
        choices_dict = choices

    # Create a dictionary mapping labels to texts
    choices_map = dict(zip(choices_dict["label"], choices_dict["text"]))

    # determine the target then normalize to letter
    answerKey = record["answerKey"]

    # return sample
    return Sample(
        input=record["question"], choices=list(choices_map.values()), target=answerKey
    )


def sample_to_fewshot(sample, prompt_template):
    choices_dict = {chr(65 + i): choice for i, choice in enumerate(sample.choices)}
    formatted_prompt = prompt_template.format(
        question=sample.input,
        a=choices_dict.get("A", ""),
        b=choices_dict.get("B", ""),
        c=choices_dict.get("C", ""),
        d=choices_dict.get("D", ""),
    )
    return f"{formatted_prompt}{sample.target}"


def arc_task(dataset_name, prompt, shot_count=0):
    plan = []

    if shot_count > 0:
        # Get few-shot examples from the train split
        fewshots = hf_dataset(
            path="ebayes/uhura-arc-easy",
            name=dataset_name,
            split="train",
            sample_fields=record_to_sample,
            shuffle=True,
            seed=42,
            limit=shot_count,
        )
        fewshot_examples = "\n\n".join(
            [sample_to_fewshot(sample, prompt) for sample in fewshots]
        )
        plan.append(
            system_message(
                f"Here are some example questions and answers:\n\n{fewshot_examples}"
            )
        )

    plan.append(system_message(prompt))
    plan.append(multiple_choice())

    return Task(
        dataset=hf_dataset(
            path="ebayes/uhura-arc-easy",
            name=dataset_name,
            split="test",
            sample_fields=record_to_sample,
        ),
        plan=plan,
        scorer=logprob_based_scorer(),
    )


@task
def amharic(prompt, shot_count=0):
    return arc_task("am_multiple_choice", prompt, shot_count)


@task
def english(prompt, shot_count=0):
    return arc_task("en_multiple_choice", prompt, shot_count)


@task
def hausa(prompt, shot_count=0):
    return arc_task("ha_multiple_choice", prompt, shot_count)


@task
def sotho(prompt, shot_count=0):
    return arc_task("nso_multiple_choice", prompt, shot_count)


@task
def swahili(prompt, shot_count=0):
    return arc_task("sw_multiple_choice", prompt, shot_count)


@task
def yoruba(prompt, shot_count=0):
    return arc_task("yo_multiple_choice", prompt, shot_count)


"""
@task
def zulu(prompt, shot_count=0):
    return arc_task("zu_multiple_choice", prompt, shot_count)
"""

if __name__ == "__main__":
    languages = [amharic, english, hausa, sotho, swahili, yoruba]
    prompts = {
        "PROMPT_1": PROMPT_1,
        "PROMPT_2": PROMPT_2,
        "PROMPT_3": PROMPT_3,
        "PROMPT_4": PROMPT_4,
        "PROMPT_5": PROMPT_5,
    }
    models = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
    ]
    shot_counts = [0, 5]

    for model in models:
        for prompt_name, prompt in prompts.items():
            for shot_count in shot_counts:
                print(
                    f"\nEvaluating with model: {model}, prompt: {prompt_name}, and {shot_count}-shot"
                )
                tasks = [lang(prompt, shot_count) for lang in languages]
                eval(tasks, model=model)
