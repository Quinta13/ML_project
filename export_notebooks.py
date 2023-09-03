import os
import subprocess
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

    # Loop through notebooks in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".ipynb"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + ".pdf")

            # Run the nbconvert command to convert the notebook to PDF
            command = [
                "jupyter",
                "nbconvert",
                "--to",
                "pdf",
                input_path,
                "--output",
                output_path,
            ]

            try:
                subprocess.run(command, check=True)
                print(f"Notebook '{input_path}' exported to PDF: '{output_path}'")
            except subprocess.CalledProcessError as e:
                print(f"Error exporting '{input_path}' to PDF: {e}")

    print("All notebooks exported to PDF.")


if __name__ == "__main__":
    export()
