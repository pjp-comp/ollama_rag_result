python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate it (Mac/Linux)
venv\Scripts\activate  # (Windows)

pip install -r requirements.txt  # Install dependencies


## Before to push project, please update package list
pip freeze > requirements.txt