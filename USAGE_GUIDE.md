# CUAD Contract Classifier - Usage Guide

## Quick Start (Zero Arguments!)

This repository provides two contract classification approaches, both requiring **ZERO command-line arguments**.

---

## 🚀 Standard Classifier (Fast & Parallel)

**Best for:** Large datasets, speed-critical applications

```bash
python cuad_processing/oss120_bedrock_labeler.py
```

### What it does:
- ✅ Processes contracts in parallel (200 threads)
- ✅ Incremental saving with resume capability
- ✅ Outputs to: `cuad_processing/output/labeled_contracts.csv`
- ✅ Fast throughput for bulk processing

### Performance:
- **Speed:** ~0.5-2 seconds per contract (parallel)
- **Accuracy:** Standard OSS120 model accuracy

---

## 🧠 Agentic Classifier (Accurate & Reflective)

**Best for:** High-accuracy requirements, critical analysis

```bash
python cuad_processing/run_agentic_classifier.py
```

### What it does:
- ✅ Uses LangGraph's 3-agent reflection pattern
- ✅ Classifier → Critic → Arbitrator workflow
- ✅ Outputs to: `cuad_processing/output/labeled_contracts_langgraph.csv`
- ✅ Higher accuracy through iterative refinement

### Performance:
- **Speed:** ~5-10 seconds per contract (sequential)
- **Accuracy:** Enhanced through multi-agent reflection

---

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install boto3 pyyaml langgraph
```

### 2. Configure AWS Credentials

Create `cuad_processing/config.yaml`:
```yaml
bedrock_region: us-east-1
bedrock_access_key: YOUR_AWS_ACCESS_KEY
bedrock_secret_access_key: YOUR_AWS_SECRET_KEY
bedrock_model_id: oss120
```

### 3. Generate Input Data

```bash
python cuad_processing/generate_cuad_csv.py
```

This creates CSV files in `cuad_processing/output/` from the CUAD dataset.

---

## 🎯 Complete Workflow

```bash
# Step 1: Generate input CSV files
python cuad_processing/generate_cuad_csv.py

# Step 2: Choose your classifier

# Option A: Fast parallel processing
python cuad_processing/oss120_bedrock_labeler.py

# Option B: Accurate agentic processing
python cuad_processing/run_agentic_classifier.py
```

---

## 📊 Output Format

Both classifiers produce CSV files with **83 columns**:

| Column | Description |
|--------|-------------|
| `contract_id` | Unique contract identifier |
| `{category}_text` | Extracted clause text (41 categories) |
| `{category}_label` | Binary label: 1 (present) or 0 (absent) |

### Example Output Structure:
```
contract_id, document_name_text, document_name_label, parties_text, parties_label, ...
CONTRACT_001, "Master Services Agreement", 1, "Company A and Company B", 1, ...
```

---

## 🔄 Resume Capability

Both classifiers support automatic resume:

- Progress is tracked in `.{filename}_progress.txt`
- If interrupted, simply re-run the same command
- Already processed contracts are automatically skipped
- No data loss or duplicate processing

---

## 🆚 Which Classifier Should I Use?

| Scenario | Recommended Classifier |
|----------|----------------------|
| Processing 1000+ contracts | **Standard** (parallel) |
| Need results quickly | **Standard** (parallel) |
| Critical legal analysis | **Agentic** (reflection) |
| Maximum accuracy required | **Agentic** (reflection) |
| Research/benchmarking | **Both** (compare results) |

---

## 🛠️ Troubleshooting

### "Configuration file not found"
- Ensure `cuad_processing/config.yaml` exists
- Check AWS credentials are valid

### "No CSV files found in output directory"
- Run `generate_cuad_csv.py` first
- Check `cuad_processing/output/` directory exists

### "LangGraph not available"
- Install: `pip install langgraph`
- Only required for agentic classifier

### Rate Limiting / Throttling
- Standard classifier: Reduce `threads` in code (default: 200)
- Agentic classifier: Already sequential, add delays if needed

---

## 📝 The 41 CUAD Categories

Both classifiers analyze contracts across these categories:

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
23. IP Ownership Assignment
24. Joint IP Ownership
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

## 🔗 Related Files

- `oss120_bedrock_labeler.py` - Standard classifier implementation
- `run_agentic_classifier.py` - Agentic classifier implementation
- `langgraph_classifier.py` - LangGraph reflection logic
- `generate_cuad_csv.py` - Input data preparation
- `config.yaml` - AWS Bedrock configuration

---

## 💡 Tips

1. **Test First:** Run on a small subset before processing entire dataset
2. **Monitor Costs:** AWS Bedrock charges per API call
3. **Compare Results:** Run both classifiers on same data to evaluate accuracy
4. **Backup Progress:** The `.progress.txt` files enable resume - don't delete them
5. **Check Logs:** Both scripts print detailed progress and error messages

---

## 📚 Additional Resources

- [CUAD Dataset Paper](https://arxiv.org/abs/2103.06268)
- [Atticus Project Website](https://www.atticusprojectai.org/cuad)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

**Questions or Issues?** Check the repository's Issues page or README.md for more information.
