# Classification of Pancreatic Ductal Carcinoma Patients using a Large Language model

### This Python script provides a command-line interface (CLI) for predicting PDAC patients based on the expression of a 5-gene biomarker using a fine-tuned ESM2 model.

 Prerequisites:
 * Python (>=3.9)
 * scikit-learn == 1.6.1
 * pandas
 * torch
 * json
 * esm == 2.0.0
 * huggingface_hub == 0.29.3

 You can install the required packages using pip:
 ```bash
 pip install pandas torch transformers esm huggingface_hub scikit
```
 #### Usage:
 `python standalone.py <filepath> [--separator <separator>] [--output <output filepath>]`
 * `<filepath>`: Path to the input file.
 * `--separator or -s`: Separator used in the input file (default: ',').
 * `--output or -o`: Path to the output file.

 #### Description:
 The script performs the following steps:
 1. Input Data Validation: Reads the input gene expression (in TPM) file using pandas, allowing the user to specify the separator.
 2. Probability Prediction: Uses pre-trained Linear Regression (LR) models (pickled files named <column_name>_5.pkl) located in the same directory as the script to predict probabilities for each specified column. The order of columns is respected.
 3. Amino Acid Conversion: Converts the predicted probabilities to amino acids based on predefined ranges.
 4. Peptide Sequence Generation: Concatenates the amino acids for each row into a peptide sequence.
 5. ESM2 Classification:
    * Downloads the fine-tuned ESM2 model (shubhamc-iiitd/pdac_pred_llm) from Hugging Face.
    * Classifies the generated peptide sequences using the ESM2 model.
    * Prints the predicted class and probability for each sequence.

 #### Model Directory:
 The script assumes that the pickled LR models (<column_name>_5.pkl) are located in the same directory as the Python script.

 #### Example:
 `python standalone.py data.csv --separator "\t" --output results/output.csv`

 This command will read data.csv, using tab as the separator, and use the columns gene1, gene2, gene3, gene4, and gene5 in that specific order.

 ####  Code Structure:
 * ```validate_file(filepath, separator)```: Validates the input file.
 * ```predict_probabilities(df, model_dir, ordered_cols)```: Predicts probabilities using LR models.
 * ```probability_to_amino_acid(probability)```: Converts probability to amino acid.
 * ```main()```: Main function to orchestrate the process.

