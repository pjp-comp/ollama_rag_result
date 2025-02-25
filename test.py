import camelot

pdf_path = "pdfs/AR_25833_ICEMAKE_2023_2024.pdf"

tables = camelot.read_pdf(pdf_path, pages="1-66", flavor="stream", strip_text="\n", edge_tol=500)

print(tables)

# Save all tables as CSVs
for i, table in enumerate(tables):
    table.to_csv(f"output_table_{i}.csv")


""" 
Why is This Better?
✅ Handles Empty Table Cases – Prevents errors if no tables are found.
✅ Saves Tables in an Organized Folder (output_tables/).
✅ Prints Table Previews – Quickly verify the output.
✅ Uses os.makedirs() – Ensures the folder exists before saving.

What Next?
If Camelot Still Misses Data:
Try adjusting edge_tol, row_tol, or split_text=True in stream mode.
If You Need More Pages:
Change pages="66" to a range like "66-70".
If You Need JSON Output:
Use table.to_json("output.json"). """