"""
Automatically execute all Docker deployment scripts and configure environment variables
"""

import os
import subprocess
import sys
import time
from contextlib import contextmanager
import json
from dotenv import load_dotenv
import argparse

sys.path.insert(0, os.getcwd())

print(f'Current directory: {os.getcwd()}')
# Define inner working directory
WORKING_DIR = "launcher"
# Configuration file path - relative to outer path
URL_CONFIG_PATH = "config/envs/webarena/init/webarena_url.json"

# Define list of scripts to execute (in order) - relative to inner path
SCRIPTS = [
    "02_docker_remove_containers.sh",
    "03_docker_create_containers.sh",
    "04_docker_start_containers.sh",
    "05_docker_patch_containers.sh"
]


def wrapper_tasks(src_dir: str, tgt_dir: str, benchmark):
    """
    Process task files by wrapping them in a standardized format

    Args:
        src_dir: Source directory containing task files
        tgt_dir: Target directory for processed files
        benchmark: Benchmark name to include in the wrapped tasks
    """
    # Ensure target directory exists
    os.makedirs(tgt_dir, exist_ok=True)

    # Traverse source directory, locate all digit.json files
    for filename in os.listdir(src_dir):
        if filename.endswith(".json") and filename[:-5].isdigit():  # Check if it's a digit.json file
            src_file_path = os.path.join(src_dir, filename)

            # Read source file content
            with open(src_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract required fields
            intent = data.get("intent", "")
            task_config = {
                "example_id": filename,
                "file_path": os.path.join(src_dir, filename).replace("\\", "/")  # Ensure path format is POSIX (cross-platform)
            }

            # Construct target file content
            tgt_data = {
                "application": None,
                "url": None,
                "query": intent,
                "task_config": task_config,
                "benchmark": benchmark
            }

            # Write target file
            tgt_file_path = os.path.join(tgt_dir, filename)
            with open(tgt_file_path, "w", encoding="utf-8") as f:
                json.dump(tgt_data, f, indent=4, ensure_ascii=False)

            print(f"Processed: {filename} -> {tgt_file_path}")

@contextmanager
def change_working_directory(new_dir):
    """Temporarily change working directory without affecting external environment"""
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(old_dir)

def load_environment():
    """Load environment variables and set them in the current Python process"""
    env_file = os.path.join(WORKING_DIR, "00_vars.sh")
    load_dotenv(env_file)

    print("Environment variables set:")
    for key, value in os.environ.items():
        print(f"{key} = {value}")


def execute_scripts():
    """Execute all scripts in sequence within the inner directory"""
    print("Starting script execution sequence...")

    # Check if inner directory exists
    if not os.path.isdir(WORKING_DIR):
        print(f"Error: Directory {WORKING_DIR} does not exist")
        sys.exit(1)

    # Switch to inner directory to execute scripts
    with change_working_directory(WORKING_DIR):
        print(f"Switched to working directory: {os.getcwd()}")

        for script in SCRIPTS:
            if not os.path.isfile(script):
                continue

            # Set execute permission
            os.chmod(script, os.stat(script).st_mode | 0o111)

            try:
                # Use subprocess to execute script and display output in real time
                process = subprocess.Popen(
                    f"./{script}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                # Display output in real time
                for line in process.stdout:
                    print(line, end='')

            except Exception as e:
                print(f"Error occurred while executing script {script}: {str(e)}")
                sys.exit(1)

            # Wait for 5 seconds between scripts
            time.sleep(5)

def write_env_variables():
    """Write environment variables to configuration file"""
    # Define environment variables to write to configuration file
    env_vars = {
        "SHOPPING_URL": os.getenv("SHOPPING"),
        "SHOPPING_ADMIN_URL": os.getenv("SHOPPING_ADMIN"),
        "REDDIT_URL": os.getenv("REDDIT"),
        "GITLAB_URL": os.getenv("GITLAB"),
        "MAP_URL": os.getenv("MAP")
    }

    try:
        with open(URL_CONFIG_PATH, 'w') as config_file:
            json.dump(env_vars, config_file, indent=4)
    except Exception as e:
        print(f"Error occurred while writing environment variables: {str(e)}")
        sys.exit(1)

def generate_test_data_for_single_set(set_name):
    """
    Generate test data for a specific set by replacing placeholders with actual URLs

    Args:
        set_name: Directory containing the test files
    """
    # Read the raw test file
    with open(os.path.join(set_name, "test.raw.json"), "r", encoding="utf-8") as f:
        raw = f.read()

    # Replace placeholders with actual environment URLs
    raw = raw.replace("__GITLAB__", os.getenv("GITLAB"))
    raw = raw.replace("__REDDIT__", os.getenv("REDDIT"))
    raw = raw.replace("__SHOPPING__", os.getenv("SHOPPING"))
    raw = raw.replace("__SHOPPING_ADMIN__", os.getenv("SHOPPING_ADMIN"))
    raw = raw.replace("__MAP__", os.getenv("MAP"))

    # Write the processed test file
    with open(os.path.join(set_name, "test.json"), "w") as f:
        f.write(raw)

    # Split into multiple files (one per test case)
    data = json.loads(raw)
    print(f'Raw length: {len(data)}')
    for idx, item in enumerate(data):
        with open(os.path.join(set_name, f"{idx}.json"), "w") as f:
            json.dump(item, f, indent=2)

def generate_test_data():
    """Generate all test data and wrap tasks in the required format"""
    # Generate test data for WebArena tasks
    generate_test_data_for_single_set("config/envs/webarena/tasks/")
    # Wrap tasks in the standardized format for the benchmark
    wrapper_tasks("config/envs/webarena/tasks/", "tasks/", "vab_webarena_lite")

def main():
    """Main execution function"""
    # Load environment variables
    load_environment()
    # Execute Docker startup scripts
    execute_scripts()
    # Persist necessary variable information
    write_env_variables()
    # Instantiate specific task information
    generate_test_data()

if __name__ == "__main__":
    """
    Main entry point:
    1. Automatically execute all Docker deployment scripts
    2. Configure environment variable files
    3. Instantiate test task files
    """
    main()