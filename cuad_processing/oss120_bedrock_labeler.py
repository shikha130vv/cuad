
"""
OSS120 Bedrock Contract Labeler

This module uses AWS Bedrock's OSS120 model to automatically label and categorize
contract text across the 41 CUAD categories.

OSS120 is a specialized model for contract analysis that can identify and extract
key clauses and provisions from legal documents.
"""

import json
import boto3
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import csv
from datetime import datetime


class OSS120ContractLabeler:
    """
    A class to label contract text using AWS Bedrock's OSS120 model.
    
    The OSS120 model is designed for contract understanding and can identify
    various legal clauses and provisions across the 41 CUAD categories.
    """
    
    # The 41 CUAD categories
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
        "Third Party Beneficiary",
        "Termination For Cause"
    ]
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the OSS120 Contract Labeler.
        
        Args:
            config_path: Path to the configuration file containing AWS credentials
        """
        self.config = self._load_config(config_path)
        self.bedrock_client = self._initialize_bedrock_client()
        self.model_id = self.config.get('bedrock_model_id', 'oss120')
        self.write_lock = threading.Lock()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                "Please create a config.yaml or config.json file with your AWS credentials."
            )
        
        with open(config_file, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError("Config file must be .yaml, .yml, or .json")
        
        return config
    

    
    def _initialize_bedrock_client(self):
        """Initialize AWS Bedrock client with credentials from config."""
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=self.config['bedrock_region'],
            aws_access_key_id=self.config['bedrock_access_key'],
            aws_secret_access_key=self.config['bedrock_secret_access_key']
        )
    
    def _create_prompt(self, contract_text: str, categories: Optional[List[str]] = None) -> str:
        """
        Create a prompt for the OSS120 model to analyze contract text and return
        STRICT, machine-parseable JSON.
        """
        if categories is None:
            categories = self.CUAD_CATEGORIES

        categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])

        prompt = f"""You are a precise legal contract analysis model.

        Analyze the following contract text and, for EVERY category in the list below, decide:
        - Does this category appear in the contract text? (1 = yes, 0 = no)
        - If yes, extract the most relevant clause text.
        - If no, use an empty string for the clause text.

        Categories:
        {categories_list}

        Contract Text:
        \"\"\"{contract_text}\"\"\"

        OUTPUT FORMAT (VERY IMPORTANT):
        - Return a SINGLE JSON object.
        - The JSON MUST be valid according to the JSON standard:
        - Use double quotes for all keys and string values.
        - Do NOT use single quotes anywhere.
        - Do NOT include comments, ellipses (...), or trailing commas.
        - Do NOT include any extra keys.
        - Do NOT wrap the JSON in markdown fences.
        - Do NOT output explanations, reasoning, or any other text outside the JSON.

        The JSON object MUST have EXACTLY one key per category, where the key is the category name
        EXACTLY as written in the list above. Each value MUST be an object with this structure:

        {{
        "Category Name": {{
            "label": 0 or 1,
            "text": "extracted clause text or empty string"
        }},
        ...
        }}

        Example of the structure (this is only an illustrative example):

        {{
        "Document Name": {{
            "label": 1,
            "text": "MASTER SERVICES AGREEMENT"
        }},
        "Parties": {{
            "label": 0,
            "text": ""
        }}
        }}

        Now produce ONLY the final JSON object, nothing else."""
        return prompt

    
    import json
    from typing import Optional, List, Dict, Any

    def label_contract(
        self, 
        contract_text: str, 
        categories: Optional[List[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Label a contract using the OSS120 model on AWS Bedrock.
        
        Args:
            contract_text: The contract text to analyze
            categories: Optional list of specific categories to analyze
            max_tokens: Maximum tokens for the response
            temperature: Temperature for model sampling (lower = more deterministic)
            
        Returns:
            Dictionary mapping category names to their labels and extracted text
        """
        prompt = self._create_prompt(contract_text, categories)

        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            # Invoke the model
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )

            # Parse Bedrock's wrapper JSON
            response_body = json.loads(response["body"].read())

            # ----------- EXTRACT MODEL TEXT PROPERLY -----------
            result_text = ""

            if "choices" in response_body:
                # OpenAI-style schema (what OSS120 is using)
                choice = response_body["choices"][0]
                message = choice.get("message", {})
                result_text = message.get("content", "") or ""
            elif "content" in response_body:
                # Claude-style
                if isinstance(response_body["content"], list) and response_body["content"]:
                    # e.g. [{"type": "text", "text": "..."}, ...]
                    block = response_body["content"][0]
                    if isinstance(block, dict):
                        result_text = block.get("text", "") or ""
                    else:
                        result_text = str(block)
                else:
                    result_text = str(response_body["content"])
            elif "completion" in response_body:
                # Some completion-style models
                result_text = response_body["completion"]
            elif "generated_text" in response_body:
                # Generic text field
                result_text = response_body["generated_text"]
            else:
                # Fallback (shouldn't be hit for OSS120 now)
                result_text = str(response_body)

            # ----------- CLEAN UP WRAPPING / REASONING -----------
            result_text = result_text.strip()

            # Strip markdown fences if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            result_text = result_text.strip()

            # Strip <reasoning> ... </reasoning> if present
            # (OSS models are clearly prepending reasoning)
            reasoning_start = result_text.find("<reasoning>")
            reasoning_end = result_text.find("</reasoning>")

            if reasoning_start != -1 and reasoning_end != -1:
                # Keep only text after </reasoning>
                result_text = result_text[reasoning_end + len("</reasoning>"):].strip()

            # ----------- EXTRACT JSON SUBSTRING -----------
            start = result_text.find("{")
            end = result_text.rfind("}")

            if start == -1 or end == -1 or end < start:
                raise ValueError(f"No JSON object found in model output:\n{result_text[:500]}")

            json_str = result_text[start:end + 1]

            # ----------- PARSE JSON -----------
            labels = json.loads(json_str)

            return labels

        except Exception as e:
            # Use result_text if it exists for debugging
            raise Exception(f"Error invoking Bedrock model: {str(e)}  Output: {result_text}")

    
    def label_contract_batch(
        self, 
        contracts: List[Dict[str, str]],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Label multiple contracts in batch (sequential processing).
        
        Args:
            contracts: List of dictionaries with 'id' and 'text' keys
            categories: Optional list of specific categories to analyze
            
        Returns:
            List of dictionaries with contract IDs and their labels
        """
        results = []
        
        for contract in contracts:
            contract_id = contract.get('id', 'unknown')
            contract_text = contract.get('text', '')
            
            try:
                labels = self.label_contract(contract_text, categories)
                results.append({
                    'contract_id': contract_id,
                    'labels': labels,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'contract_id': contract_id,
                    'labels': {},
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _process_single_contract(
        self,
        contract: Dict[str, str],
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a single contract (used by parallel processing).
        
        Args:
            contract: Dictionary with 'id' and 'text' keys
            categories: Optional list of specific categories to analyze
            
        Returns:
            Dictionary with contract ID, labels, and status
        """
        contract_id = contract.get('id', 'unknown')
        contract_text = contract.get('text', '')
        
        try:
            labels = self.label_contract(contract_text, categories)
            return {
                'contract_id': contract_id,
                'labels': labels,
                'status': 'success'
            }
        except Exception as e:
            print(f"ERROR processing contract {contract_id}: {str(e)}")
            return {
                'contract_id': contract_id,
                'labels': {},
                'status': 'error',
                'error': str(e)
            }
    
    def label_contract_batch_parallel(
        self,
        contracts: List[Dict[str, str]],
        output_path: str,
        categories: Optional[List[str]] = None,
        max_workers: int = 200,
        resume: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Label multiple contracts in parallel with incremental persistence and resume capability.
        
        Args:
            contracts: List of dictionaries with 'id' and 'text' keys
            output_path: Path to save the CSV file (used for incremental writes)
            categories: Optional list of specific categories to analyze
            max_workers: Number of parallel threads (default: 200)
            resume: Whether to skip already processed contracts (default: True)
            
        Returns:
            List of dictionaries with contract IDs and their labels
        """
        output_file = Path(output_path)
        progress_file = output_file.parent / f".{output_file.stem}_progress.txt"
        
        # Track processed contracts
        processed_ids = set()
        if resume and progress_file.exists():
            with open(progress_file, 'r') as f:
                processed_ids = set(line.strip() for line in f if line.strip())
            print(f"Resume mode: Found {len(processed_ids)} already processed contracts")
        
        # Filter out already processed contracts
        contracts_to_process = [
            c for c in contracts 
            if c.get('id', 'unknown') not in processed_ids
        ]
        
        if not contracts_to_process:
            print("All contracts already processed!")
            return []
        
        print(f"Processing {len(contracts_to_process)} contracts with {max_workers} threads...")
        
        # Initialize CSV file with header if it doesn't exist
        if not output_file.exists():
            self._initialize_csv(output_path)
        
        results = []
        completed_count = 0
        total_count = len(contracts_to_process)
        
        # Process contracts in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_contract = {
                executor.submit(self._process_single_contract, contract, categories): contract
                for contract in contracts_to_process
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_contract):
                contract = future_to_contract[future]
                contract_id = contract.get('id', 'unknown')
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Only persist successful results
                    if result['status'] == 'success':
                        self._append_result_to_csv(result, output_path)
                        
                        # Mark as processed
                        with self.write_lock:
                            with open(progress_file, 'a') as f:
                                f.write(f"{contract_id}\n")
                    
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == total_count:
                        print(f"Progress: {completed_count}/{total_count} contracts processed")
                        
                except Exception as e:
                    print(f"ERROR: Unexpected exception for contract {contract_id}: {str(e)}")
                    results.append({
                        'contract_id': contract_id,
                        'labels': {},
                        'status': 'error',
                        'error': str(e)
                    })
        
        return results
    
    def _initialize_csv(self, output_path: str):
        """Initialize CSV file with header."""
        header = ['contract_id']
        for category in self.CUAD_CATEGORIES:
            col_name = category.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
            header.extend([f'{col_name}_text', f'{col_name}_label'])
        
        with self.write_lock:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def _append_result_to_csv(self, result: Dict[str, Any], output_path: str):
        """Append a single result to the CSV file (thread-safe)."""
        if result['status'] != 'success':
            return
        
        row = [result['contract_id']]
        labels = result['labels']
        
        for category in self.CUAD_CATEGORIES:
            category_data = labels.get(category, {'label': 0, 'text': ''})
            row.append(category_data.get('text', ''))
            row.append(category_data.get('label', 0))
        
        with self.write_lock:
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    
    def export_to_csv(
        self, 
        labeled_contracts: List[Dict[str, Any]], 
        output_path: str
    ):
        """
        Export labeled contracts to CSV format matching CUAD structure.
        
        Args:
            labeled_contracts: List of labeled contract results
            output_path: Path to save the CSV file
        """
        # Create header with contract_id + 82 columns (41 categories × 2)
        header = ['contract_id']
        for category in self.CUAD_CATEGORIES:
            # Normalize category name for column
            col_name = category.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
            header.extend([f'{col_name}_text', f'{col_name}_label'])
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for contract in labeled_contracts:
                if contract['status'] != 'success':
                    continue
                
                row = [contract['contract_id']]
                labels = contract['labels']
                
                for category in self.CUAD_CATEGORIES:
                    category_data = labels.get(category, {'label': 0, 'text': ''})
                    row.append(category_data.get('text', ''))
                    row.append(category_data.get('label', 0))
                
                writer.writerow(row)


def load_contracts_from_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Load contracts from CSV file generated by generate_cuad_csv.py.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of dictionaries with 'id' and 'text' keys for processing
    """
    contracts = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            contract_id = row.get('contract_id', 'unknown')
            
            # Extract the full contract text from all text columns
            # The CSV has columns like: category_text, category_label
            # We'll concatenate all non-empty text fields to reconstruct the contract
            text_parts = []
            for key, value in row.items():
                if key.endswith('_text') and value and value.strip():
                    text_parts.append(value)
            
            # If no text parts found, skip this contract
            if not text_parts:
                continue
            
            contract_text = ' '.join(text_parts)
            
            contracts.append({
                'id': contract_id,
                'text': contract_text
            })
    
    return contracts


def main():
    """Example usage of the OSS120ContractLabeler."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='Label contracts using AWS Bedrock OSS120 model with parallel processing'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./cuad_processing/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input contract text file, JSON file, or CSV file with contracts'
    )
    parser.add_argument(
        '--input-csv',
        type=str,
        help='Input CSV file from generate_cuad_csv.py (alternative to --input)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./cuad_processing/output',
        help='Directory containing CSV files to process (will process all CSV files in this directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--process-output-folder',
        action='store_true',
        help='Process all CSV files from the output folder generated by generate_cuad_csv.py'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=200,
        help='Number of parallel threads for processing (default: 200)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume capability (start from scratch)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Enable parallel processing (default: True)'
    )
    
    args = parser.parse_args()
    
    # Initialize labeler
    labeler = OSS120ContractLabeler(config_path=args.config)
    
    contracts = []
    output_path = args.output
    
    # Determine input source
    if args.process_output_folder:
        # Process all CSV files in the output directory
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Error: Output directory {output_dir} does not exist.")
            print("Please run generate_cuad_csv.py first to generate CSV files.")
            return
        
        csv_files = list(output_dir.glob('*.csv'))
        if not csv_files:
            print(f"No CSV files found in {output_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV file(s) in {output_dir}")
        
        for csv_file in csv_files:
            print(f"\nProcessing {csv_file.name}...")
            file_contracts = load_contracts_from_csv(str(csv_file))
            contracts.extend(file_contracts)
            print(f"  Loaded {len(file_contracts)} contracts from {csv_file.name}")
        
        # Set default output path if not specified
        if not output_path:
            output_path = str(output_dir / 'labeled_contracts.csv')
    
    elif args.input_csv:
        # Process specific CSV file
        print(f"Loading contracts from CSV: {args.input_csv}")
        contracts = load_contracts_from_csv(args.input_csv)
        print(f"Loaded {len(contracts)} contracts")
        
        if not output_path:
            input_path = Path(args.input_csv)
            output_path = str(input_path.parent / f'labeled_{input_path.name}')
    
    elif args.input:
        # Original functionality: process JSON or text file
        input_path = Path(args.input)
        
        if input_path.suffix == '.csv':
            # Handle CSV input
            contracts = load_contracts_from_csv(str(input_path))
        elif input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                contracts = json.load(f)
        else:
            # Assume plain text file
            with open(input_path, 'r') as f:
                contract_text = f.read()
            contracts = [{'id': input_path.stem, 'text': contract_text}]
        
        if not output_path:
            output_path = str(input_path.parent / f'labeled_{input_path.stem}.csv')
    
    else:
        # Default behavior: look for CSV files in the output folder
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            csv_files = list(output_dir.glob('*.csv'))
            if csv_files:
                print(f"No input specified. Found {len(csv_files)} CSV file(s) in {output_dir}")
                print("Processing all CSV files from output folder...")
                
                for csv_file in csv_files:
                    print(f"\nProcessing {csv_file.name}...")
                    file_contracts = load_contracts_from_csv(str(csv_file))
                    contracts.extend(file_contracts)
                    print(f"  Loaded {len(file_contracts)} contracts from {csv_file.name}")
                
                if not output_path:
                    output_path = str(output_dir / 'labeled_contracts.csv')
            else:
                print(f"Error: No CSV files found in {output_dir}")
                print("Please specify --input, --input-csv, or run generate_cuad_csv.py first.")
                return
        else:
            print("Error: No input specified and output directory does not exist.")
            print("Usage:")
            print("  --input <file>              : Process a single text/JSON/CSV file")
            print("  --input-csv <file>          : Process a specific CSV file from generate_cuad_csv.py")
            print("  --process-output-folder     : Process all CSV files in the output directory")
            print("  --output-dir <dir>          : Specify output directory (default: ./cuad_processing/output)")
            print("  --threads <n>               : Number of parallel threads (default: 200)")
            print("  --no-resume                 : Disable resume capability")
            return
    
    if not contracts:
        print("No contracts to process.")
        return
    
    # Label contracts
    print(f"\nLabeling {len(contracts)} contract(s) using AWS Bedrock OSS120 model...")
    print(f"Parallel processing: {args.parallel}, Threads: {args.threads}, Resume: {not args.no_resume}")
    
    start_time = datetime.now()
    
    if args.parallel:
        # Use parallel processing with incremental persistence
        results = labeler.label_contract_batch_parallel(
            contracts,
            output_path,
            max_workers=args.threads,
            resume=not args.no_resume
        )
        print(f"\n✓ Results incrementally saved to {output_path}")
    else:
        # Use sequential processing
        results = labeler.label_contract_batch(contracts)
        labeler.export_to_csv(results, output_path)
        print(f"\n✓ Results saved to {output_path}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"  Total contracts: {len(contracts)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Duration: {duration:.2f} seconds")
    if success_count > 0:
        print(f"  Average time per contract: {duration/len(contracts):.2f} seconds")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
