"""
CUAD Dataset CSV Generator

This script processes the CUAD (Contract Understanding Atticus Dataset) and generates
a CSV file with 83 columns:
- 1 contract_id column
- 41 categories, each with 2 columns (text and label) = 82 columns

The 41 categories from CUAD include various contract clauses such as:
Document Name, Parties, Agreement Date, Effective Date, Expiration Date,
Renewal Term, Notice Period To Terminate Renewal, Governing Law, 
Most Favored Nation, Non-Compete, Exclusivity, No-Solicit Of Customers,
No-Solicit Of Employees, Non-Disparagement, Termination For Convenience,
Rofr/Rofo/Rofn, Change Of Control, Anti-Assignment, Revenue/Profit Sharing,
Price Restrictions, Minimum Commitment, Volume Restriction, Ip Ownership Assignment,
Joint Ip Ownership, License Grant, Non-Transferable License, Affiliate License-Licensor,
Affiliate License-Licensee, Unlimited/All-You-Can-Eat-License, Irrevocable Or Perpetual License,
Source Code Escrow, Post-Termination Services, Audit Rights, Uncapped Liability,
Cap On Liability, Liquidated Damages, Warranty Duration, Insurance, Covenant Not To Sue,
Third Party Beneficiary
"""

import json
import csv
import os
from typing import Dict, List, Any
import zipfile


# Define all 41 CUAD categories
CUAD_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary"
]


def normalize_category_name(category: str) -> str:
    """Normalize category name for column headers."""
    return category.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def extract_data_from_json(json_file_path: str) -> List[Dict[str, Any]]:
    """
    Extract contract data from CUAD JSON format.
    
    Args:
        json_file_path: Path to the JSON file containing CUAD data
        
    Returns:
        List of dictionaries containing processed contract data
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    contracts = []
    
    # Process each contract in the dataset
    i_item = 0
    for item in data.get('data', []):
        i_item += 1
        i_para = 0
        for paragraph in item.get('paragraphs', []):
            i_para += 1
            contract_id = paragraph.get('context_id', "item_" + str(i_item) + "_para_" + str(i_para))
            contract_text = paragraph.get('context', '')
            
            # Initialize contract record
            contract_record = {
                'contract_id': contract_id,
                'full_text': contract_text
            }
            
            # Initialize all categories with empty values
            for category in CUAD_CATEGORIES:
                normalized_cat = normalize_category_name(category)
                contract_record[f'{normalized_cat}_text'] = ''
                contract_record[f'{normalized_cat}_label'] = 0
            
            # Process QA pairs to extract category information
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
                
                # Match question to category
                for category in CUAD_CATEGORIES:
                    if category.lower() in question.lower():
                        normalized_cat = normalize_category_name(category)
                        
                        # Extract answers
                        answers = qa.get('answers', [])
                        if answers and len(answers) > 0:
                            # Combine multiple answers with separator
                            answer_texts = [ans.get('text', '') for ans in answers if ans.get('text', '')]
                            if answer_texts:
                                contract_record[f'{normalized_cat}_text'] = ' | '.join(answer_texts)
                                contract_record[f'{normalized_cat}_label'] = 1
                        
                        # Check is_impossible flag
                        if qa.get('is_impossible', False):
                            contract_record[f'{normalized_cat}_label'] = 0
                        
                        break
            
            contracts.append(contract_record)
    
    return contracts


def generate_csv(input_json_path: str, output_csv_path: str):
    """
    Generate CSV file from CUAD JSON data.
    
    Args:
        input_json_path: Path to input JSON file
        output_csv_path: Path to output CSV file
    """
    print(f"Processing {input_json_path}...")
    
    # Extract data
    contracts = extract_data_from_json(input_json_path)
    
    if not contracts:
        print("No contracts found in the dataset!")
        return
    
    # Define CSV columns (83 total)
    columns = ['contract_id']
    for category in CUAD_CATEGORIES:
        normalized_cat = normalize_category_name(category)
        columns.append(f'{normalized_cat}_text')
        columns.append(f'{normalized_cat}_label')
    
    # Write to CSV
    print(f"Writing {len(contracts)} contracts to {output_csv_path}...")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(contracts)
    
    print(f"✓ Successfully created CSV with {len(columns)} columns and {len(contracts)} rows")
    print(f"  Columns: 1 contract_id + {len(CUAD_CATEGORIES)} categories × 2 (text + label) = {len(columns)} total")


def process_cuad_dataset(data_dir: str = './data', output_dir: str = './output'):
    """
    Process the complete CUAD dataset and generate CSV files.
    
    Args:
        data_dir: Directory containing CUAD data files
        output_dir: Directory to save output CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if data.zip exists and extract it
    if os.path.exists('data.zip') and not os.path.exists(data_dir):
        print("Extracting data.zip...")
        with zipfile.ZipFile('data.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    # Process train, dev, and test sets
    for dataset_type in ['train', 'dev', 'test']:
        json_file = os.path.join(data_dir, f'CUADv1_{dataset_type}.json')
        
        if os.path.exists(json_file):
            output_csv = os.path.join(output_dir, f'cuad_{dataset_type}.csv')
            generate_csv(json_file, output_csv)
        else:
            print(f"Warning: {json_file} not found, skipping...")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate CSV from CUAD dataset')
    parser.add_argument('--data-dir', default='./cuad_processing/data', help='Directory containing CUAD JSON files')
    parser.add_argument('--output-dir', default='./cuad_processing/output', help='Directory to save CSV files')
    parser.add_argument('--input-json', default="cuad_processing/data/CUADv1.json", help='Process a single JSON file')
    parser.add_argument('--output-csv', default="cuad_processing/output/cuad.csv", help='Output CSV file path (for single file processing)')
    
    args = parser.parse_args()
    
    if args.input_json and args.output_csv:
        # Process single file
        generate_csv(args.input_json, args.output_csv)
    else:
        # Process entire dataset
        process_cuad_dataset(args.data_dir, args.output_dir)
