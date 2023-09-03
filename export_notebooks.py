import os
from os import path

import nbformat
from nbconvert import PDFExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import TagRemovePreprocessor
from nbconvert.preprocessors import ClearOutputPreprocessor

from io_ import get_root_dir

def export():

    # Directory paths
    input_directory = path.join(get_root_dir(), "notebooks")
    output_directory = path.join(get_root_dir(), "notebooks_exported")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize a PDF exporter
    pdf_exporter = PDFExporter()

    # Initialize preprocessors
    execute_preprocessor = ExecutePreprocessor(timeout=None)
    tag_remove_preprocessor = TagRemovePreprocessor(tags=["remove-cell"])
    clear_output_preprocessor = ClearOutputPreprocessor()

    # Loop through notebooks in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".ipynb"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + ".pdf")

            # Create a notebook object from the input file
            with open(input_path, "r", encoding="utf-8") as nb_file:
                notebook = nbformat.read(nb_file, as_version=4)

            # Preprocess the notebook
            notebook, _ = execute_preprocessor.preprocess(notebook, {})
            notebook, _ = tag_remove_preprocessor.preprocess(notebook, {})
            notebook, _ = clear_output_preprocessor.preprocess(notebook, {})

            # Export the notebook to PDF
            pdf_data, _ = pdf_exporter.from_notebook_node(notebook)

            # Write the PDF data to the output file
            with open(output_path, "wb") as pdf_file:
                pdf_file.write(pdf_data)

            print(f"Notebook exported to PDF: {output_path}")

    print("All notebooks exported to PDF.")


if __name__ == "__main__":

    export()
