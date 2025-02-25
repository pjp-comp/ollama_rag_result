import os
import sys
from ollama import ListResponse, list
import subprocess
import json
import shutil
import utils
import pprint

# Load configuration and functions from the utils
config = utils.load_config()

# print(config)
# Remove functions from the config before printing
config_copy = {key: value for key, value in config.items() if not callable(value)}

# Print the config in a formatted way
# print(json.dumps(config_copy, indent=4))


# Calculate dynamic paths
model_path = utils.get_absolute_path(os.path.join(config['model_folder'], config['model_name']))
trained_model_path = utils.get_absolute_path(os.path.join(config['trained_model_folder'], config['model_name'] + "-trained"))
training_data_path = utils.get_absolute_path(config['training_data_path'])

# print(training_data_path)

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
    downloaded_models = list_out_models()
    for eachmodel in downloaded_models:
        if(eachmodel.model == config['model_name']):
            model = run_ollama(f"ollama run {model_path}")
            return model

def list_out_models():
    response: ListResponse = list()
    pprint.pprint(response, compact=True)
    return response.models
    for model in response.models:
        print('Name:', model.model)
        print('  Size (MB):', f'{(model.size.real / 1024 / 1024):.2f}')
    if model.details:
        print('  Format:', model.details.format)
        print('  Family:', model.details.family)
        print('  Parameter Size:', model.details.parameter_size)
        print('  Quantization Level:', model.details.quantization_level)
    print('\n')
    
# Function to train the model
def train_model():
    print("Loading the model...")
    # Here, load the model from the path defined above (using the downloaded model)
    print(f"Model loaded from {model_path}")
    
    print("Training the model...")
    # Add the code for training (this is a placeholder for training logic)
    # You should implement your own training loop depending on how the model is trained.
    output = "Training output..."  # Replace with actual model training output
    
    # Placeholder for actual model training logic
    print(output)
    return output

# Function to save the trained model
def save_model():
    print(f"Saving the model to {trained_model_path}...")
    os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
    
    # Here, save the trained model to the path
    shutil.copytree(model_path, trained_model_path)
    
    print(f"Model saved at {trained_model_path}")

# Main function
def main():
    list_out_models()  # Download the model
    # load_model()  # Train the model


if __name__ == "__main__":
    main()
