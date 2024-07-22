# to run this script, run inspect eval arc_5_shot.py@amharic --model hf/meta-llama/Meta-Llama-3-8B-Instruct in the cli
# to change the language, change the @amharic to @english, @swahili, @hausa, @northern_sotho, @yoruba

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import answer, scorer, accuracy, bootstrap_std
from inspect_ai.solver import multiple_choice, system_message, TaskState
from inspect_ai.scorer._target import Target  
from inspect_ai.scorer._metric import Score  

few_shot = 5 # you can change this to any number you want

def sample_to_fewshot(sample):
    choices_text = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample.choices)])
    return f"Question: {sample.input}\n\nChoices:\n{choices_text}\n\nAnswer: {sample.target}"

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

        if logprobs is None:
            return Score(
                value="C" if is_correct else "I",
                answer=model_answer,
                explanation=f"Model answer: {model_answer}, Correct answer: {correct_answer}, No logprobs available"
            )

        avg_logprob = sum(lp.logprob for lp in logprobs.content) / len(logprobs.content)

        return Score(
            value="C" if is_correct else "I",
            answer=model_answer,
            explanation=f"Model answer: {model_answer}, Correct answer: {correct_answer}, Average log probability: {avg_logprob:.4f}",
            metadata={"avg_logprob": avg_logprob}
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
    input=record["question"],
    choices=list(choices_map.values()),
    target=answerKey
  )

def arc_task(dataset_name):
    # Get 5 fewshot examples from the train split
    fewshots = hf_dataset(
        path="ebayes/uhura-arc-easy",
        name=dataset_name,
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=few_shot,
    )
    
    fewshot_examples = "\n\n".join([sample_to_fewshot(sample) for sample in fewshots])
    
    return Task(
        dataset=hf_dataset(
            path="ebayes/uhura-arc-easy",
            name=dataset_name,
            split="test",
            sample_fields=record_to_sample
        ),
        plan = [
            system_message(f"Here are some example questions and answers:\n\n{fewshot_examples}"),
            multiple_choice()
        ],
        scorer = logprob_based_scorer()  
    )
   
@task
def amharic():
  return arc_task("am_multiple_choice")

@task
def english():
  return arc_task("en_multiple_choice")

@task
def swahili():
  return arc_task("sw_multiple_choice")

@task
def hausa():
  return arc_task("ha_multiple_choice")

@task
def yoruba():
  return arc_task("yo_multiple_choice")