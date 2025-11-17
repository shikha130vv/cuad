# Contract Understanding Atticus Dataset

This repository contains code for the [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad), pronounced "kwad", a dataset for legal contract review curated by the Atticus Project. It is part of the associated paper [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268) by [Dan Hendrycks](http://danhendrycks.com/), [Collin Burns](http://collinpburns.com), Anya Chen, and Spencer Ball.

Contract review is a task about "finding needles in a haystack." 
We find that Transformer models have nascent performance on CUAD, but that this performance is strongly influenced by model design and training dataset size. Despite some promising results, there is still substantial room for improvement. As one of the only large, specialized NLP benchmarks annotated by experts, CUAD can serve as a challenging research benchmark for the broader NLP community.

<img align="center" src="contract_review.png" width="1000">

For more details about CUAD and legal contract review, see the [Atticus Project website](https://www.atticusprojectai.org/cuad).

## Trained Models

We [provide checkpoints](https://zenodo.org/record/4599830) for three of the best models fine-tuned on CUAD: RoBERTa-base (~100M parameters), RoBERTa-large (~300M parameters), and DeBERTa-xlarge (~900M parameters). 

## Extra Data
Researchers may be interested in several gigabytes of unlabeled contract pretraining data, which is available [here](https://drive.google.com/file/d/1of37X0hAhECQ3BN_004D8gm6V88tgZaB/view?usp=sharing).

## Requirements

This repository requires the HuggingFace [Transformers](https://huggingface.co/transformers) library. It was tested with Python 3.8, PyTorch 1.7, and Transformers 4.3/4.4. 

## AWS Bedrock OSS120 Contract Labeling

This repository includes tools for automated contract labeling using AWS Bedrock's OSS120 model. Two approaches are available:

### 1. Standard Classifier (Fast, Parallel Processing)

The standard classifier uses direct OSS120 model inference with parallel processing for high throughput.

**Run with ZERO arguments:**
```bash
python cuad_processing/oss120_bedrock_labeler.py
```

**Features:**
- ✓ Parallel processing with 200 threads
- ✓ Incremental persistence and resume capability
- ✓ Outputs to `labeled_contracts.csv`
- ✓ Fast processing for large datasets

### 2. Agentic Classifier (Accurate, LangGraph Reflection)

The agentic classifier uses LangGraph's reflection pattern with a three-agent system for higher accuracy.

**Run with ZERO arguments:**
```bash
python cuad_processing/run_agentic_classifier.py
```

**Features:**
- ✓ Three-agent reflection: Classifier → Critic → Arbitrator
- ✓ Higher accuracy through iterative refinement
- ✓ Outputs to `labeled_contracts_langgraph.csv`
- ✓ Sequential processing (slower but more accurate)

### Setup

1. Install dependencies:
```bash
pip install boto3 pyyaml langgraph
```

2. Configure AWS credentials in `cuad_processing/config.yaml`:
```yaml
bedrock_region: us-east-1
bedrock_access_key: YOUR_ACCESS_KEY
bedrock_secret_access_key: YOUR_SECRET_KEY
bedrock_model_id: oss120
```

3. Generate input CSV files:
```bash
python cuad_processing/generate_cuad_csv.py
```

4. Run your preferred classifier (no arguments needed!):
```bash
# Standard classifier (fast)
python cuad_processing/oss120_bedrock_labeler.py

# OR

# Agentic classifier (accurate)
python cuad_processing/run_agentic_classifier.py
```

### Output Format

Both classifiers produce CSV files with 83 columns:
- `contract_id`: Unique identifier
- For each of the 41 CUAD categories:
  - `{category}_text`: Extracted clause text
  - `{category}_label`: Binary label (0 or 1)

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2021cuad,
          title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review}, 
          author={Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
          journal={NeurIPS},
          year={2021}
    }
