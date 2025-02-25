
# Load configuration and handle all logic
# def load_config():
#     # Load the config file
#     with open("config.json", "r") as config_file:
#         config = json.load(config_file)

#     # Get the absolute path based on the relative model path in the configuration
#     def get_absolute_path(relative_path):
#         project_path = os.path.dirname(os.path.realpath(__file__))  # Current project folder
#         return os.path.join(project_path, relative_path)

#     # Run ollama commands
#     def run_ollama(command):
#         result = subprocess.run(command, shell=True, capture_output=True, text=True)
#         return result.stdout
    
#     # Attach absolute path function to the config
#     config['get_absolute_path'] = get_absolute_path
#     config['run_ollama'] = run_ollama

#     return config


# lib.py

import json
import os
import subprocess

# Function to load configuration from config.json
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

# Utility to print the configuration in a formatted way, excluding functions
def print_config(config):
    config_copy = {key: value for key, value in config.items() if not callable(value)}
    print(json.dumps(config_copy, indent=4))

