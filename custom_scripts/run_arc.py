import subprocess
import csv
import os
import json
import re

# List of languages to evaluate
languages = ["amharic", "hausa", "sotho", "swahili", "yoruba", "english", "zulu"]

# List of models to evaluate
models = [
    "google/gemini-1.5-flash",
]

# List of few_shot values to evaluate
few_shots = [0, 5]  # 0 for zero-shot, 5 for five-shot etc

# List of prompt IDs to evaluate
prompt_ids = [1, 2, 3, 4, 5]

output_file = "output.csv"

# Regex pattern to extract the log file path from the command output
log_pattern = re.compile(r"Log: (.*\.json)")

# Open the CSV file for writing
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(["Language", "Model", "Few-Shot", "Prompt_ID", "Accuracy"])
    csvfile.flush()  # Flush after writing the header

    for model in models:
        for language in languages:
            for few_shot_value in few_shots:
                for prompt_id in prompt_ids:
                    # Construct the command
                    cmd = ["inspect", "eval", f"run.py@{language}", "--model", model]
                    print(
                        f"Running command: {' '.join(cmd)} with FEW_SHOT={few_shot_value}, PROMPT_ID={prompt_id}"
                    )

                    try:
                        # Set the FEW_SHOT and PROMPT_ID environment variables
                        env = os.environ.copy()
                        env["FEW_SHOT"] = str(few_shot_value)
                        env["PROMPT_ID"] = str(prompt_id)

                        # Run the command and capture the output
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            env=env,
                        )

                        if result.returncode != 0:
                            print(
                                f"Command failed for {language} on model {model} with FEW_SHOT={few_shot_value}, PROMPT_ID={prompt_id}: {result.stderr}"
                            )
                            writer.writerow(
                                [
                                    language,
                                    model,
                                    few_shot_value,
                                    prompt_id,
                                    "Command failed",
                                ]
                            )
                            continue

                        # Extract the log file path from the command output
                        log_match = log_pattern.search(result.stdout)
                        if not log_match:
                            print(
                                f"Log file not found in output for {language} on model {model} with FEW_SHOT={few_shot_value}, PROMPT_ID={prompt_id}"
                            )
                            writer.writerow(
                                [
                                    language,
                                    model,
                                    few_shot_value,
                                    prompt_id,
                                    "Log file not found",
                                ]
                            )
                            continue

                        log_file_path = log_match.group(1).strip()
                        print(
                            f"Log file for {language} on model {model} with FEW_SHOT={few_shot_value}, PROMPT_ID={prompt_id}: {log_file_path}"
                        )

                        # Read the log file and extract accuracy
                        if not os.path.isfile(log_file_path):
                            print(
                                f"Log file does not exist for {language} on model {model} with FEW_SHOT={few_shot_value}, PROMPT_ID={prompt_id}: {log_file_path}"
                            )
                            writer.writerow(
                                [
                                    language,
                                    model,
                                    few_shot_value,
                                    prompt_id,
                                    "Log file does not exist",
                                ]
                            )
                            continue

                        with open(log_file_path, "r", encoding="utf-8") as f:
                            log_data = json.load(f)

                        # Extract accuracy from scores list
                        results = log_data.get("results", {})
                        scores = results.get("scores", [])

                        accuracy_value = None

                        # Iterate over scores to find the accuracy metric
                        for score_entry in scores:
                            metrics = score_entry.get("metrics", {})
                            accuracy_metric = metrics.get("accuracy", {})
                            if "value" in accuracy_metric:
                                accuracy_value = accuracy_metric["value"]
                                break  # Stop after finding the first accuracy value

                        if accuracy_value is not None:
                            writer.writerow(
                                [
                                    language,
                                    model,
                                    few_shot_value,
                                    prompt_id,
                                    accuracy_value,
                                ]
                            )
                            csvfile.flush()
                            print(
                                f"Completed evaluation for {language} on model {model} with FEW_SHOT={few_shot_value}, PROMPT_ID={prompt_id}: Accuracy = {accuracy_value}"
                            )
                        else:
                            print(
                                f"Accuracy not found in log for {language} on model {model} with FEW_SHOT={few_shot_value}, PROMPT_ID={prompt_id}"
                            )
                            writer.writerow(
                                [
                                    language,
                                    model,
                                    few_shot_value,
                                    prompt_id,
                                    "Accuracy not found",
                                ]
                            )

                    except Exception as e:
                        print(
                            f"An error occurred for {language} on model {model} with FEW_SHOT={few_shot_value}, PROMPT_ID={prompt_id}: {e}"
                        )
                        writer.writerow(
                            [language, model, few_shot_value, prompt_id, f"Error: {e}"]
                        )
