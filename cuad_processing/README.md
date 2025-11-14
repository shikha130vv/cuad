
# CUAD Processing Tools

This directory contains tools for processing and analyzing the CUAD (Contract Understanding Atticus Dataset) using both traditional data processing and AI-powered contract labeling.

## 📁 Directory Contents

- **`generate_cuad_csv.py`** - Script to generate structured CSV files from CUAD JSON data
- **`oss120_bedrock_labeler.py`** - AI-powered contract labeling using AWS Bedrock's OSS120 model
- **`CSV_GENERATION_README.md`** - Detailed documentation for CSV generation
- **`config.yaml`** / **`config.json`** - Configuration templates for AWS Bedrock credentials

## 🎯 Overview

### The 41 CUAD Categories

Both tools work with the following 41 contract clause categories:

1. Document Name
2. Parties
3. Agreement Date
4. Effective Date
5. Expiration Date
6. Renewal Term
7. Notice Period To Terminate Renewal
8. Governing Law
9. Most Favored Nation
10. Non-Compete
11. Exclusivity
12. No-Solicit Of Customers
13. No-Solicit Of Employees
14. Non-Disparagement
15. Termination For Convenience
16. Rofr/Rofo/Rofn
17. Change Of Control
18. Anti-Assignment
19. Revenue/Profit Sharing
20. Price Restrictions
21. Minimum Commitment
22. Volume Restriction
23. Ip Ownership Assignment
24. Joint Ip Ownership
25. License Grant
26. Non-Transferable License
27. Affiliate License-Licensor
28. Affiliate License-Licensee
29. Unlimited/All-You-Can-Eat-License
30. Irrevocable Or Perpetual License
31. Source Code Escrow
32. Post-Termination Services
33. Audit Rights
34. Uncapped Liability
35. Cap On Liability
36. Liquidated Damages
37. Warranty Duration
38. Insurance
39. Covenant Not To Sue
40. Third Party Beneficiary
41. Termination For Cause

---

## 🔧 Tool 1: CSV Generation from CUAD Dataset

### Purpose
Convert CUAD JSON files into structured CSV format with 83 columns (1 contract_id + 82 columns for 41 categories × 2).

### Usage

#### Process the entire CUAD dataset:
```bash
python generate_cuad_csv.py --data-dir ./data --output-dir ./output
```

This generates three CSV files:
- `cuad_train.csv`
- `cuad_dev.csv`
- `cuad_test.csv`

#### Process a single JSON file:
```bash
python generate_cuad_csv.py --input-json ./data/CUADv1_train.json --output-csv ./output/cuad_train.csv
```

### Requirements
- Python 3.6+
- Standard library only (no external dependencies)

### Output Format
- **83 columns**: `contract_id` + 41 categories × 2 (text + label)
- **UTF-8 encoding**
- **Pipe-separated** (`|`) for multiple clauses in same category
- **Binary labels**: 1 if clause exists, 0 if not

For more details, see [`CSV_GENERATION_README.md`](./CSV_GENERATION_README.md).

---

## 🤖 Tool 2: AI-Powered Contract Labeling with OSS120

### Purpose
Use AWS Bedrock's OSS120 model to automatically label and categorize contract text across the 41 CUAD categories using advanced AI.

### Setup

#### 1. Configure AWS Credentials

Create a configuration file with your AWS Bedrock credentials. You can use either YAML or JSON format:

**Option A: config.yaml**
```yaml
aws_access_key_id: "YOUR_AWS_ACCESS_KEY_ID"
aws_secret_access_key: "YOUR_AWS_SECRET_ACCESS_KEY"
aws_region: "us-east-1"
model_id: "oss120"
```

**Option B: config.json**
```json
{
  "aws_access_key_id": "YOUR_AWS_ACCESS_KEY_ID",
  "aws_secret_access_key": "YOUR_AWS_SECRET_ACCESS_KEY",
  "aws_region": "us-east-1",
  "model_id": "oss120"
}
```

**⚠️ SECURITY WARNING**: Never commit your config file to version control! Add it to `.gitignore`:
```bash
echo "config.yaml" >> ../.gitignore
echo "config.json" >> ../.gitignore
```

#### 2. Install Dependencies
```bash
pip install boto3 pyyaml
```

### Usage

#### Command Line Interface

Label a single contract text file:
```bash
python oss120_bedrock_labeler.py --config config.yaml --input contract.txt --output results.csv
```

Label multiple contracts from JSON:
```bash
python oss120_bedrock_labeler.py --config config.yaml --input contracts.json --output results.csv
```

#### Python API

```python
from oss120_bedrock_labeler import OSS120ContractLabeler

# Initialize the labeler
labeler = OSS120ContractLabeler(config_path='config.yaml')

# Label a single contract
contract_text = "This Agreement is entered into..."
labels = labeler.label_contract(contract_text)

# Label multiple contracts
contracts = [
    {'id': 'contract_001', 'text': 'Contract text 1...'},
    {'id': 'contract_002', 'text': 'Contract text 2...'}
]
results = labeler.label_contract_batch(contracts)

# Export to CSV
labeler.export_to_csv(results, 'output.csv')
```

### Features

- **Automatic clause extraction**: Identifies and extracts relevant text for each category
- **Binary classification**: Determines presence (1) or absence (0) of each clause type
- **Batch processing**: Process multiple contracts efficiently
- **CSV export**: Output in CUAD-compatible format (83 columns)
- **Configurable parameters**: Adjust temperature, max_tokens, etc.

### Requirements
- Python 3.7+
- `boto3` - AWS SDK for Python
- `pyyaml` - YAML configuration support
- AWS Bedrock access with OSS120 model enabled

### AWS Bedrock Setup

1. **Create AWS Account**: Sign up at [aws.amazon.com](https://aws.amazon.com)
2. **Enable Bedrock**: Navigate to AWS Bedrock in your AWS Console
3. **Request Model Access**: Request access to the OSS120 model
4. **Create IAM User**: Create an IAM user with Bedrock permissions
5. **Generate Credentials**: Create access keys for the IAM user

Required IAM permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## 📊 Comparison: CSV Generation vs. OSS120 Labeling

| Feature | CSV Generation | OSS120 Labeling |
|---------|---------------|-----------------|
| **Input** | CUAD JSON files | Any contract text |
| **Method** | Data transformation | AI-powered analysis |
| **Speed** | Very fast | Moderate (API calls) |
| **Accuracy** | 100% (existing labels) | High (AI-based) |
| **Use Case** | Process existing CUAD data | Label new contracts |
| **Dependencies** | None | boto3, AWS Bedrock |
| **Cost** | Free | AWS Bedrock pricing |

---

## 🚀 Quick Start Guide

### Scenario 1: Working with Existing CUAD Data
```bash
# Generate CSV from CUAD dataset
python generate_cuad_csv.py --data-dir ../data --output-dir ./output
```

### Scenario 2: Labeling New Contracts
```bash
# 1. Set up AWS credentials in config.yaml
# 2. Install dependencies
pip install boto3 pyyaml

# 3. Label your contracts
python oss120_bedrock_labeler.py --config config.yaml --input new_contract.txt --output labeled.csv
```

---

## 📝 Output Format

Both tools produce CSV files with the same structure:

```
contract_id, document_name_text, document_name_label, parties_text, parties_label, ...
```

- **83 columns total**
- **1 contract per row**
- **Text columns**: Extracted clause text (empty if not present)
- **Label columns**: Binary 0/1 indicating presence

---

## 🔒 Security Best Practices

1. **Never commit credentials**: Add `config.yaml` and `config.json` to `.gitignore`
2. **Use IAM roles**: When running on AWS infrastructure, use IAM roles instead of access keys
3. **Rotate credentials**: Regularly rotate your AWS access keys
4. **Limit permissions**: Grant only necessary Bedrock permissions
5. **Encrypt at rest**: Store config files in encrypted storage

---

## 🐛 Troubleshooting

### CSV Generation Issues
- **File not found**: Ensure data.zip is extracted to `../data/` directory
- **Encoding errors**: CUAD data is UTF-8 encoded

### OSS120 Labeling Issues
- **Authentication error**: Verify AWS credentials in config file
- **Model not found**: Ensure OSS120 model access is enabled in AWS Bedrock
- **Rate limiting**: Implement exponential backoff for batch processing
- **Timeout errors**: Increase `timeout_seconds` in config

---

## 📚 Additional Resources

- [CUAD Dataset Paper](https://arxiv.org/abs/2103.06268)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Original CUAD Repository](https://github.com/TheAtticusProject/cuad)

---

## 📄 License

This code is provided as part of the CUAD project. Please refer to the main repository license.

---

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Add tests for new features
- Update documentation accordingly
- Never commit AWS credentials

---

## 📧 Support

For issues related to:
- **CSV Generation**: Check `CSV_GENERATION_README.md`
- **AWS Bedrock**: Consult AWS Bedrock documentation
- **CUAD Dataset**: Visit the original CUAD repository

---

**Last Updated**: November 2025
