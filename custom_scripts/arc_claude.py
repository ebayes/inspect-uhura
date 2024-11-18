# to run this script, run python arc_claude.py in the cli
# to change the language, change the @amharic to @english, @swahili, @hausa, @northern_sotho, @yoruba

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import answer, scorer, accuracy, bootstrap_std
from inspect_ai.solver import multiple_choice, system_message, TaskState
from inspect_ai.scorer._target import Target
from inspect_ai.scorer._metric import Score
from prompts import PROMPT_1, PROMPT_2, PROMPT_3, PROMPT_4, PROMPT_5
import json
from pathlib import Path
import csv
import os
import glob
from datetime import datetime


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
            path="uhuradata/uhura-arc-easy",
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
            path="uhuradata/uhura-arc-easy",
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


@task
def zulu(prompt, shot_count=0):
    return arc_task("zu_multiple_choice", prompt, shot_count)


if __name__ == "__main__":
    languages = [zulu]  # amharic, english, hausa, sotho, swahili, yoruba,
    language_names = {
        # amharic: "amharic",
        # english: "english",
        # hausa: "hausa",
        # sotho: "sotho",
        # swahili: "swahili",
        # yoruba: "yoruba",
        zulu: "zulu",
    }
    prompts = {
        "PROMPT_1": PROMPT_1,
        "PROMPT_2": PROMPT_2,
        "PROMPT_3": PROMPT_3,
        "PROMPT_4": PROMPT_4,
        "PROMPT_5": PROMPT_5,
    }
    models = [
        "anthropic/claude-3-5-sonnet-20241022",
    ]
    shot_counts = [0]

    # Create or open the CSV file
    csv_filename = "arc_claude_results.csv"
    csv_exists = os.path.isfile(csv_filename)

    try:
        with open(csv_filename, "a", newline="") as csvfile:
            fieldnames = [
                "Model",
                "Template",
                "Shot Count",
                "Language",
                "Accuracy",
                "Bootstrap STD",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not csv_exists:
                writer.writeheader()

            for model in models:
                for prompt_name, prompt in prompts.items():
                    for shot_count in shot_counts:
                        for lang in languages:
                            if lang not in language_names:
                                print(
                                    f"Error: Language {lang.__name__} not found in language_names dictionary."
                                )
                                continue

                            language_name = language_names[lang]

                            try:
                                tasks = [lang(prompt, shot_count)]
                                results = eval(tasks, model=model)

                                # Find the most recent log file
                                list_of_files = glob.glob("./logs/*.json")
                                if not list_of_files:
                                    print("Error: No log files found.")
                                    continue

                                latest_file = max(list_of_files, key=os.path.getctime)

                                with open(latest_file, "r") as f:
                                    log_data = json.load(f)

                                scores = log_data["results"]["scores"][0]["metrics"]
                                accuracy = scores["accuracy"]["value"]
                                bootstrap_std = scores["bootstrap_std"]["value"]

                                # Write the results to the CSV file
                                writer.writerow(
                                    {
                                        "Model": model,
                                        "Template": prompt_name,
                                        "Shot Count": shot_count,
                                        "Language": language_name,
                                        "Accuracy": accuracy,
                                        "Bootstrap STD": bootstrap_std,
                                    }
                                )

                            except Exception as e:
                                print(f"Error processing {language_name}: {str(e)}")

        print(f"Results have been saved to {csv_filename}")
    except IOError as e:
        print(f"Error opening or writing to CSV file: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
