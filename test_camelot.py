import os
import camelot

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

pdf_path = "./pdfs/AR_25833_ICEMAKE_2023_2024.pdf"  # Replace with an actual PDF file
tables = camelot.read_pdf(pdf_path, pages="1")
print(tables)

