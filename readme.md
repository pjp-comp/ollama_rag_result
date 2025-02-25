### Create Virtual env
clone project

cd ollama_ai
python -m venv venv  
source venv/bin/activate  
source venv\Scripts\activate (mac)  

pip install -r requirements.txt  # Install dependencies


### Before to push project, please update package list
pip freeze > requirements.txt

if found any issues try installing pdfplumber without dependencies: 
pip install pdfplumber --no-deps


1. For vectorDB : ChromaDB
2. PDF Table Extraction using Camelot