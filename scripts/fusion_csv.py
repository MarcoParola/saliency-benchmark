import pandas as pd
import os

from src.utils import retrieve_concepts_for_class

dataset_name = 'intel_image'

def process_folder(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    combined_df = []
    print(all_files)

    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, index_col=0)
        df_transposed = df.T

        df = df_transposed
        # Add new column for WOE Score
        df["WOE Score"] = 0.0

        for label, row in df.iterrows():
            print(f"Processing row with label: {label}")
            favor_list, against_list = retrieve_concepts_for_class(label, dataset_name)

            woe_score = 0
            for col in df.columns[:-1]:  # Exclude the last column (WOE Score)
                print(f"Processing column {col}")
                value = row[col]
                if col in favor_list:
                    print(f"Favor: {value}")
                    woe_score += value * 1
                elif col in against_list:
                    print(f"Against: {value}")
                    woe_score += value * -1
                else:
                    print(f"exclused: {value}")
                    woe_score += value * 0

            df.at[label, "WOE Score"] = woe_score

        # Tokenize file name to extract metadata
        tokens = file_name.replace(".csv", "").split("_")
        extractor = tokens[1] if len(tokens) > 1 else ""
        classifier = tokens[3] if len(tokens) > 3 else ""
        backbone = tokens[4] if len(tokens) > 4 else ""
        concept_presence = tokens[5] if len(tokens) > 5 else ""

        # Add extracted metadata as new columns
        df.insert(0, "Saliency", extractor)
        df.insert(1, "Classifier", classifier)
        df.insert(2, "Extractor", backbone)
        df.insert(3, "Concept Presence", concept_presence)

        combined_df.append(df_transposed)
    # Concatenate all processed dataframes
    final_df = pd.concat(combined_df)
    return final_df

# Example usage
if __name__ == "__main__":
    path = os.path.join(os.path.abspath("ablation_results"),'fusion_results')
    output_df = process_folder(os.path.join(path,dataset_name))
    output_df.to_csv(os.path.join(path,f"fusion_output_{dataset_name}.csv"), index=True)

