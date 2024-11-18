import subprocess
import json
import os
import csv
from datetime import datetime

lang_codes = ["yo", "am", "ha", "nso", "sw", "en", "zu"]
few_shots = [0, 5]
models = [
    "google/gemini-1.5-flash",
]

# Create or open the CSV file for appending
csv_file = "truthfulqa_results_claude.csv"
csv_exists = os.path.isfile(csv_file)


def print_accuracy_and_bootstrap(data):
    results = data.get("results", {})
    scores = results.get("scores", [])

    for score in scores:
        if score.get("name") == "logprob_based_scorer":
            metrics = score.get("metrics", {})
            accuracy = metrics.get("accuracy", {}).get("value")
            bootstrap_std = metrics.get("bootstrap_std", {}).get("value")

            if accuracy is not None and bootstrap_std is not None:
                print(
                    f"Debug: Found accuracy: {accuracy}, bootstrap_std: {bootstrap_std}"
                )
                return accuracy, bootstrap_std

    print("Debug: Data structure:")
    print(json.dumps(data, indent=2))
    return None, None


with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow(
            ["Timestamp", "Language", "Few-shot", "Model", "Accuracy", "Bootstrap Std"]
        )
        print("Debug: CSV header written")

    for lang in lang_codes:
        for shots in few_shots:
            for model in models:
                print(f"\nDebug: Processing {lang}, {shots} shots, {model}")

                # Modify the original script
                with open("final_truthfulqa_5.py", "r") as script_file:
                    content = script_file.read()

                modified_content = content.replace(
                    'lang_code = "yo"', f'lang_code = "{lang}"'
                ).replace("few_shot = 5", f"few_shot = {shots}")

                with open("temp_truthfulqa.py", "w") as temp_file:
                    temp_file.write(modified_content)

                # Run the modified script
                command = f"inspect eval temp_truthfulqa.py --model {model}"
                print(f"Running: {command}")
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"Error running command: {result.stderr}")
                    continue

                # Find the latest JSON file in the logs directory
                log_files = [f for f in os.listdir("./logs") if f.endswith(".json")]
                if not log_files:
                    print("No JSON files found in ./logs directory")
                    continue
                latest_log = max(
                    log_files, key=lambda f: os.path.getmtime(os.path.join("./logs", f))
                )
                print(f"Debug: Latest log file: {latest_log}")

                # Read and parse the JSON file
                with open(os.path.join("./logs", latest_log), "r") as json_file:
                    data = json.load(json_file)

                # Extract accuracy and bootstrap_std using the new function
                accuracy, bootstrap_std = print_accuracy_and_bootstrap(data)

                print(
                    f"Debug: Extracted accuracy = {accuracy}, bootstrap_std = {bootstrap_std}"
                )

                if accuracy is not None and bootstrap_std is not None:
                    # Write to CSV
                    timestamp = datetime.now().isoformat()
                    row = [timestamp, lang, shots, model, accuracy, bootstrap_std]
                    writer.writerow(row)
                    f.flush()  # Ensure data is written to the file
                    print(f"Debug: Row written to CSV: {row}")
                else:
                    print(
                        f"Accuracy and Bootstrap STD not found for {lang}, {shots} shots, {model}"
                    )

# Clean up temporary file
subprocess.run("rm temp_truthfulqa.py", shell=True)
print("Debug: Script completed")
