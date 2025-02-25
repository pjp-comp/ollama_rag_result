import json
import subprocess
import os

# Load configuration from config.json
def load_config():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    return config

# Function to get the absolute path based on the project path
def get_absolute_path(relative_path):
    project_path = os.path.dirname(os.path.realpath(__file__))  # Current project folder
    return os.path.join(project_path, relative_path)

# Function to run ollama commands in the background
def run_ollama(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

# Load the configuration
config = load_config()

# Calculate dynamic paths
model_path = get_absolute_path(os.path.join(config['model_folder'], config['model_name']))
trained_model_path = get_absolute_path(os.path.join(config['trained_model_folder'], config['model_name'] + "-trained"))
training_data_path = get_absolute_path(config['training_data_file'])

# Pull the model (if not already done)
def download_model():
    print("Downloading model...")
    output = run_ollama(f"ollama pull {config['model_name']}")
    print(output)
    
# def run_ollama(command):
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     print(f"Command: {command}")  # Print the command being run for debugging
#     print(f"Stdout: {result.stdout}")  # Print the standard output
#     print(f"Stderr: {result.stderr}")  # Print any error output
#     return result.stdout

def run_ollama(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, timeout=60)
        print(f"Command: {command}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except subprocess.TimeoutExpired as e:
        print(f"Command timed out: {e}")

# Load the model and start working with it
def load_model():
    print("Loading the model...")
    model = run_ollama(f"ollama run {model_path}")
    return model

# Train the model
def train_model():
    print("Training the model...")
    # Use the dynamically calculated training data path
    training_command = f"ollama train {model_path} --data {training_data_path}"
    output = run_ollama(training_command)
    print("Training output:", output)

# Save the trained model
def save_model():
    print(f"Saving the model to {trained_model_path}...")
    save_command = f"ollama save {model_path} --path {trained_model_path}"
    output = run_ollama(save_command)
    print(f"Save output: {output}")

    # Check if the model has been saved by listing files in the trained_models folder
    if os.path.exists(trained_model_path):
        print(f"Model saved successfully to {trained_model_path}")
    else:
        print(f"Error: Model not saved to {trained_model_path}")


# Example of using the functions
# download_model()
# model = load_model()
# train_model()
# save_model()
