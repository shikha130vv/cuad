# CUAD Dataset CSV Generator

This directory contains code to generate a structured CSV file from the CUAD (Contract Understanding Atticus Dataset).

## Overview

The script `generate_cuad_csv.py` processes the CUAD dataset and creates a CSV file with **83 columns**:

- **1 column**: `contract_id` - Unique identifier for each contract
- **82 columns**: 41 categories × 2 columns each (text and label)

## The 41 CUAD Categories

The script extracts the following contract clause categories:

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

## Column Structure

For each category, two columns are created:
- `{category}_text`: The extracted text/clause content
- `{category}_label`: Binary label (1 if clause exists, 0 if not)

## Usage

### Process the entire CUAD dataset:

```bash
python generate_cuad_csv.py --data-dir ./data --output-dir ./output
```

This will generate three CSV files:
- `cuad_train.csv`
- `cuad_dev.csv`
- `cuad_test.csv`

### Process a single JSON file:

```bash
python generate_cuad_csv.py --input-json ./data/CUADv1_train.json --output-csv ./output/cuad_train.csv
```

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)

## Data Format

The script expects CUAD data in JSON format (as provided in the original dataset). The data.zip file should be extracted to a `data/` directory containing:
- `CUADv1_train.json`
- `CUADv1_dev.json`
- `CUADv1_test.json`

## Output

The generated CSV files will have:
- **83 columns** total
- One row per contract
- UTF-8 encoding
- Pipe-separated (`|`) multiple clauses within a single category

## Notes

- Multiple clauses for the same category are concatenated with ` | ` separator
- Empty categories have empty text and label = 0
- The script handles the CUAD v1 format with QA pairs
