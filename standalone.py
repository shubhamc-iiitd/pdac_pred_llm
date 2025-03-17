import argparse
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
import esm
from huggingface_hub import hf_hub_download
import json

def validate_file(filepath, separator):
    """Validates if the file exists and can be read with the given separator."""
    try:
        df = pd.read_csv(filepath, sep=separator)
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty: {filepath}")
        return None
    except pd.errors.ParserError:
        print(f"Error: File has parsing errors with separator '{separator}': {filepath}")
        return None

def predict_probabilities(df, model_dir, ordered_cols):
    """Predicts probabilities using LR models for each column."""
    results = []
    for col in ordered_cols:
        model_path = os.path.join(model_dir, f"{col}_5.pkl")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            probabilities = model.predict_proba(df[[col]])[:, 1]  # Get probability of class 1
            results.append(pd.Series(probabilities, name=col))
        except FileNotFoundError:
            print(f"Error: Model file not found for column {col}: {model_path}")
            return None
        except Exception as e:
            print(f"Error during prediction for column {col}: {e}")
            return None

    return pd.concat(results, axis=1)

def probability_to_amino_acid(probability):
    """Converts probability to amino acid based on given ranges."""
    if 0 <= probability < 0.05:
        return 'A'
    elif 0.05 <= probability < 0.1:
        return 'C'
    elif 0.1 <= probability < 0.15:
        return 'D'
    elif 0.15 <= probability < 0.2:
        return 'E'
    elif 0.2 <= probability < 0.25:
        return 'F'
    elif 0.25 <= probability < 0.3:
        return 'G'
    elif 0.3 <= probability < 0.35:
        return 'H'
    elif 0.35 <= probability < 0.4:
        return 'I'
    elif 0.4 <= probability < 0.45:
        return 'K'
    elif 0.45 <= probability < 0.5:
        return 'L'
    elif 0.5 <= probability < 0.55:
        return 'M'
    elif 0.55 <= probability < 0.6:
        return 'N'
    elif 0.6 <= probability < 0.65:
        return 'P'
    elif 0.65 <= probability < 0.7:
        return 'Q'
    elif 0.7 <= probability < 0.75:
        return 'R'
    elif 0.75 <= probability < 0.8:
        return 'S'
    elif 0.8 <= probability < 0.85:
        return 'T'
    elif 0.85 <= probability < 0.9:
        return 'V'
    elif 0.9 <= probability < 0.95:
        return 'W'
    elif 0.95 <= probability <= 1.0:
        return 'Y'
    else:
        return 'X' # For values outside the 0-1 range

def main():
    """Main function to parse command-line arguments, validate file, predict probabilities, and generate peptide sequences."""
    parser = argparse.ArgumentParser(description="Predict probabilities, convert to amino acids, generate peptide sequences, and predict using ESM2.")
    parser.add_argument("filepath", help="Path to the input file.")
    parser.add_argument("--separator", "-s", default=",", help="Separator used in the input file (default: ',').")

    args = parser.parse_args()

    # Model directory is assumed to be in the same path as the script
    model_dir = os.path.dirname(os.path.abspath(__file__))

    df = validate_file(args.filepath, args.separator)

    if df is not None:
        required_columns = ['ENSG00000204287', 'ENSG00000104894', 'ENSG00000081059', 'ENSG00000171345', 'ENSG00000265972']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Input file does not contain all required columns: {required_columns}")
            exit(1)

        probabilities_df = predict_probabilities(df, model_dir, required_columns)

        if probabilities_df is not None:
            amino_acid_df = probabilities_df.applymap(probability_to_amino_acid)
            peptide_sequences = amino_acid_df.apply(lambda row: ''.join(row), axis=1).tolist() # get python list

            # ESM2 Model Loading and Prediction
            repo_id = "shubhamc-iiitd/pdac_pred_llm"
            model_weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            alphabet_path = hf_hub_download(repo_id=repo_id, filename="alphabet.bin")
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

