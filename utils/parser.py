import pandas as pd
import numpy as np
import os

def parsing_static_files(paths):
    """
    Parses multiple Excel files and extracts all distance matrices from each sheet.
    Args:
        paths (list): List of file paths to the Excel files.
    Returns:
        list: List of distance matrices (as pandas DataFrames).
    """
    distance_matrices = []

    for file_path in paths:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load the Excel file and parse all sheets
        try:
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                # Parse the sheet and add to the list
                matrix = excel_file.parse(sheet_name=sheet_name, header=None)
                distance_matrices.append(matrix)
                print(f"Parsed sheet '{sheet_name}' from file '{file_path}'")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return distance_matrices

def parse_instance(file_path,sheet_name):
    """
    Parses an Excel file and extracts the distance matrix from a specific sheet.
    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to parse.
    """
    distance_matrix = []
    try:
        excel_file = pd.ExcelFile(file_path)
        distance_matrix = excel_file.parse(sheet_name=sheet_name, header=None).values
        print(f"Parsed sheet '{sheet_name}' from file '{file_path}'")
        #np.fill_diagonal(distance_matrix, 0)
        #print(distance_matrix, "type", type(distance_matrix), "shape", distance_matrix.shape)
        return distance_matrix
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")