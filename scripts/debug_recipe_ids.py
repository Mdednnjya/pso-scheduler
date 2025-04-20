import os
import json
import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Debug Recipe IDs')
    parser.add_argument('--input', type=str, default='output/enriched_recipes.json', 
                        help='Input JSON file with recipe data')
    parser.add_argument('--model', type=str, default='models/meal_data.json',
                        help='Model JSON file with recipe data')
    parser.add_argument('--id', type=str, help='Recipe ID to check')
    args = parser.parse_args()
    
    # Check input file
    if os.path.exists(args.input):
        print(f"\nChecking input file: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                print(f"Successfully loaded JSON with {len(data)} recipes")
                
                # Check the format of IDs
                id_examples = [item['ID'] for item in data[:5]]
                print(f"Sample IDs: {id_examples}")
                
                # Count unique IDs
                unique_ids = set(item['ID'] for item in data)
                print(f"Number of unique IDs: {len(unique_ids)}")
                
                # Check if the specified ID exists
                if args.id:
                    matching = [item for item in data if str(item['ID']) == str(args.id)]
                    if matching:
                        print(f"\nFound recipe with ID '{args.id}':")
                        print(f"Title: {matching[0]['Title']}")
                    else:
                        print(f"\nNo recipe found with ID '{args.id}'")
                        
                        # Try to find similar IDs
                        close_ids = [item['ID'] for item in data if str(args.id) in str(item['ID'])]
                        if close_ids:
                            print(f"Similar IDs found: {close_ids[:10]}")
            except json.JSONDecodeError:
                print(f"Error: The file {args.input} is not valid JSON")
    else:
        print(f"Error: Input file {args.input} not found")
    
    # Check model file
    if os.path.exists(args.model):
        print(f"\nChecking model file: {args.model}")
        with open(args.model, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                print(f"Successfully loaded JSON with {len(data)} recipes")
                
                # Check the format of IDs
                id_examples = [item['ID'] for item in data[:5]]
                print(f"Sample IDs: {id_examples}")
                
                # Check if the specified ID exists
                if args.id:
                    matching = [item for item in data if str(item['ID']) == str(args.id)]
                    if matching:
                        print(f"\nFound recipe with ID '{args.id}' in model data")
                    else:
                        print(f"\nNo recipe found with ID '{args.id}' in model data")
            except json.JSONDecodeError:
                print(f"Error: The file {args.model} is not valid JSON")
    else:
        print(f"\nModel file {args.model} not found. Has the model been trained?")
        
    # Suggest next steps
    print("\nSuggested next steps:")
    print("1. Make sure to run the training step first: python scripts/run_cbf_pipeline.py --train")
    print("2. Try using one of the sample IDs shown above in your recommendation command")
    print("3. Check for any errors in the training process")

if __name__ == "__main__":
    main()