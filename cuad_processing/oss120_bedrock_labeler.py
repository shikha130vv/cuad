
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
        self.model_id = self.config.get('model_id', 'oss120')
        
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
            region_name=self.config['aws_region'],
            aws_access_key_id=self.config['aws_access_key_id'],
            aws_secret_access_key=self.config['aws_secret_access_key']
        )
    
    def _create_prompt(self, contract_text: str, categories: Optional[List[str]] = None) -> str:
        """
        Create a prompt for the OSS120 model to analyze contract text.
        
        Args:
            contract_text: The contract text to analyze
            categories: Optional list of specific categories to focus on
            
        Returns:
            Formatted prompt string
        """
        if categories is None:
            categories = self.CUAD_CATEGORIES
        
        categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
        
        prompt = f"""You are a legal contract analysis expert. Analyze the following contract text and identify which of these categories are present:

{categories_list}

For each category found, extract the relevant text/clause and indicate whether it exists (1) or not (0).

Contract Text:
{contract_text}

Please provide your analysis in the following JSON format:
{{
    "category_name": {{
        "label": 0 or 1,
        "text": "extracted clause text or empty string"
    }},
    ...
}}

Respond only with the JSON object, no additional text."""
        
        return prompt
    
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
        
        # Prepare the request body for Bedrock
        request_body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
        
        try:
            # Invoke the model
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Extract the generated text
            if 'completion' in response_body:
                result_text = response_body['completion']
            elif 'generated_text' in response_body:
                result_text = response_body['generated_text']
            else:
                result_text = str(response_body)
            
            # Parse the JSON response from the model
            # Remove any markdown code blocks if present
            result_text = result_text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            labels = json.loads(result_text)
            return labels
            
        except Exception as e:
            raise Exception(f"Error invoking Bedrock model: {str(e)}")
    
    def label_contract_batch(
        self, 
        contracts: List[Dict[str, str]],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Label multiple contracts in batch.
        
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
        import csv
        
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


def main():
    """Example usage of the OSS120ContractLabeler."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Label contracts using AWS Bedrock OSS120 model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input contract text file or JSON file with contracts'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Initialize labeler
    labeler = OSS120ContractLabeler(config_path=args.config)
    
    # Load input
    input_path = Path(args.input)
    if input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            contracts = json.load(f)
    else:
        # Assume plain text file
        with open(input_path, 'r') as f:
            contract_text = f.read()
        contracts = [{'id': input_path.stem, 'text': contract_text}]
    
    # Label contracts
    print(f"Labeling {len(contracts)} contract(s)...")
    results = labeler.label_contract_batch(contracts)
    
    # Export to CSV
    labeler.export_to_csv(results, args.output)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
