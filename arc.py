# to run this script, run inspect eval arc.py@amharic --model hf/meta-llama/Meta-Llama-3-8B-Instruct in the cli
# to change the language, change the @amharic to @english, @swahili, @hausa, @northern_sotho, @yoruba

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import answer, scorer, accuracy, bootstrap_std
from inspect_ai.solver import multiple_choice, system_message, TaskState
from inspect_ai.scorer._target import Target  
from inspect_ai.scorer._metric import Score  

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
   return Task(
     dataset=hf_dataset(
       path="ebayes/uhura-arc-easy",
       name=dataset_name,
       split="test",
       sample_fields=record_to_sample
     ),
     plan = [
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
def sotho():
  return arc_task("nso_multiple_choice")

@task
def swahili():
  return arc_task("sw_multiple_choice")

@task
def hausa():
  return arc_task("ha_multiple_choice")

@task
def yoruba():
  return arc_task("yo_multiple_choice")