import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def read_csv_with_fallback_encodings(file_path):
    """Read CSV file with multiple encodings if needed"""
    encodings = ['utf-8', 'iso-8859-1', 'Windows-1252', 'latin1']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue

    # Last resort
    return pd.read_csv(file_path, encoding='utf-8', errors='replace')


def download_and_prepare_dataset(output_dir="data/raw", force_download=True):
    """Download datasets from Kaggle and prepare raw_combined_dataset.csv"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define dataset files in specific order
    dataset_files = [
        "dataset-udang.csv",  # 1
        "dataset-tempe.csv",  # 2
        "dataset-ayam.csv",  # 3
        "dataset-sapi.csv",  # 4
        "dataset-telur.csv",  # 5
        "dataset-tahu.csv",  # 6
        "dataset-ikan.csv",  # 7
        "dataset-kambing.csv",  # 8
    ]

    print("Downloading datasets from Kaggle...")
    # Download if needed
    if force_download or not all(os.path.exists(os.path.join(output_dir, f"raw_{f}")) for f in dataset_files):
        # Clean existing files if needed
        if force_download:
            for file_path in dataset_files:
                raw_path = os.path.join(output_dir, f"raw_{file_path}")
                if os.path.exists(raw_path):
                    os.remove(raw_path)

        # Download and extract
        api.dataset_download_files(
            "canggih/indonesian-food-recipes",
            path=output_dir,
            unzip=True
        )

        # Rename files to our convention
        for file_path in dataset_files:
            src_path = os.path.join(output_dir, file_path)
            dst_path = os.path.join(output_dir, f"raw_{file_path}")
            if os.path.exists(src_path):
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                os.rename(src_path, dst_path)

    print("Processing datasets...")
    all_datasets = []
    for i, file_path in enumerate(dataset_files):
        raw_file_path = os.path.join(output_dir, f"raw_{file_path}")

        if os.path.exists(raw_file_path):
            try:
                # Read CSV
                df = read_csv_with_fallback_encodings(raw_file_path)

                # Find love column
                love_col = next((col for col in df.columns if col.lower() in ['love', 'loves']), None)

                # Select top 10 or first 10 rows
                if love_col:
                    df[love_col] = pd.to_numeric(df[love_col], errors='coerce')
                    top_df = df.nlargest(10, love_col)
                else:
                    top_df = df.head(10)

                # Keep only required columns
                keep_cols = []
                for col in ['Title', 'Ingredients']:
                    if col in df.columns:
                        keep_cols.append(col)

                if keep_cols:
                    all_datasets.append(top_df[keep_cols])
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    # Combine all datasets
    if all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)

        # Add ID column
        combined_df.insert(0, 'ID', range(1, len(combined_df) + 1))

        # Save combined dataset
        combined_path = os.path.join(output_dir, "raw_combined_dataset.csv")
        combined_df.to_csv(combined_path, index=False, encoding='utf-8')

        # Print summary
        print("\nDataset Summary:")
        print(f"Total records: {len(combined_df)}")

        # Print category ranges
        start_idx = 1
        for i, file_path in enumerate(dataset_files):
            category = file_path.replace('dataset-', '').replace('.csv', '')
            record_count = min(10, len(all_datasets[i]))
            end_idx = start_idx + record_count - 1
            print(f"{i + 1}. {category.capitalize()} (ID: {start_idx}-{end_idx})")
            start_idx = end_idx + 1

        print(f"\nSuccessfully saved combined dataset to {combined_path}")
        return combined_path
    else:
        print("No datasets were processed successfully.")
        return None


if __name__ == "__main__":
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        import pip

        pip.main(['install', 'kaggle'])
        import kaggle

    download_and_prepare_dataset(force_download=True)