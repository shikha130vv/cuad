
# LangGraph-Based Agentic Classifier for CUAD

## Overview

This implementation provides a **LangGraph-based agentic classifier** using the **reflection design pattern** for CUAD (Contract Understanding Atticus Dataset) contract labeling. The system employs three specialized agents that work together to produce high-quality contract classifications.

## Architecture

### Reflection Design Pattern

The reflection pattern is an advanced prompting strategy that improves AI agent performance by incorporating iterative self-critique and refinement. Our implementation uses a **three-agent system**:

```
┌─────────────┐      ┌─────────┐      ┌─────────────┐
│ Classifier  │ ───> │  Critic │ ───> │ Arbitrator  │ ───> Final Result
└─────────────┘      └─────────┘      └─────────────┘
```

### Agent Roles

#### 1. **Classifier Agent**
- **Role**: Makes initial classification of contract clauses
- **Expertise**: Contract law, legal terminology, clause identification
- **Output**: Initial classification with confidence scores for all 41 CUAD categories
- **Prompt**: Focuses on thorough analysis and completeness

#### 2. **Critic Agent**
- **Role**: Reviews and critiques the classifier's output
- **Expertise**: Quality assurance, error detection, edge case identification
- **Output**: Detailed critique identifying:
  - False positives (incorrectly marked as present)
  - False negatives (missed categories)
  - Extraction issues (incomplete or inaccurate clause text)
  - Confidence concerns (low-confidence classifications)
  - Ambiguities (unclear contract language)
- **Prompt**: Focuses on finding errors and providing constructive feedback

#### 3. **Arbitrator Agent**
- **Role**: Makes final authoritative decision
- **Expertise**: Senior-level legal expertise, conflict resolution, definitive judgment
- **Output**: Final classification with decision rationale for each category
- **Prompt**: Considers both classifier and critic outputs, makes independent judgment

### Workflow

The system uses **LangGraph's StateGraph** to orchestrate a linear workflow:

1. **Classifier Node**: Analyzes contract text and produces initial classification
2. **Critic Node**: Reviews classifier output and identifies potential issues
3. **Arbitrator Node**: Makes final decision considering all inputs
4. **END**: Returns final classification

The state flows through the graph, accumulating outputs from each agent.

## Implementation Details

### State Management

The workflow uses a `LGClauseState` TypedDict to manage state:

```python
class LGClauseState(TypedDict):
    # Input
    contract_id: str
    contract_text: str
    categories: List[str]
    
    # Agent outputs
    classifier_output: Optional[Dict[str, Any]]
    critic_output: Optional[Dict[str, Any]]
    arbitrator_output: Optional[Dict[str, Any]]
    
    # Final result
    final_classification: Optional[Dict[str, Any]]
    
    # Metadata
    status: str
    error: Optional[str]
```

### Key Files

- **`langgraph_classifier.py`**: Main LangGraph processor
  - `LangGraphContractClassifier` class
  - Three agent nodes (classifier, critic, arbitrator)
  - StateGraph workflow construction
  - Bedrock API integration

- **`langgraph_prompts.py`**: Prompt templates
  - System prompts for each agent
  - Prompt builder functions
  - JSON extraction utilities

- **`oss120_bedrock_labeler.py`**: Modified to support LangGraph
  - Added `--use-langgraph` flag
  - Integration with LangGraph classifier

## Usage

### Basic Usage

```python
from langgraph_classifier import LangGraphContractClassifier
import boto3

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id='YOUR_KEY',
    aws_secret_access_key='YOUR_SECRET'
)

# Create classifier
classifier = LangGraphContractClassifier(
    bedrock_client=bedrock_client,
    model_id='oss120'
)

# Classify a contract
result = classifier.classify_contract(
    contract_text="YOUR CONTRACT TEXT HERE",
    contract_id="contract_001"
)

# Access final classification
final_labels = result['final_classification']
```

### Command-Line Usage

Use the modified `oss120_bedrock_labeler.py` with the `--use-langgraph` flag:

```bash
# Process a single contract with LangGraph
python oss120_bedrock_labeler.py \
    --input contract.txt \
    --output labeled_output.csv \
    --use-langgraph

# Process CSV files with LangGraph
python oss120_bedrock_labeler.py \
    --input-csv contracts.csv \
    --output labeled_contracts.csv \
    --use-langgraph

# Process with parallel execution (if supported)
python oss120_bedrock_labeler.py \
    --process-output-folder \
    --output-dir ./output \
    --use-langgraph \
    --threads 50
```

### Configuration

The LangGraph classifier uses the same configuration as the standard labeler:

```yaml
# config.yaml
bedrock_region: us-east-1
bedrock_access_key: YOUR_ACCESS_KEY
bedrock_secret_access_key: YOUR_SECRET_KEY
bedrock_model_id: oss120
```

## Differences from Standard Classifier

| Feature | Standard Classifier | LangGraph Classifier |
|---------|-------------------|---------------------|
| **Approach** | Single-pass classification | Three-agent reflection |
| **Quality Control** | None (direct output) | Built-in critique and review |
| **Confidence** | Not tracked | Tracked at each stage |
| **Error Detection** | Manual review needed | Automatic critic review |
| **Decision Rationale** | Not provided | Included for each category |
| **Processing Time** | Faster (1 API call) | Slower (3 API calls) |
| **Accuracy** | Good | Better (with reflection) |
| **Cost** | Lower | Higher (3x API calls) |

## Benefits of Reflection Pattern

1. **Improved Accuracy**: Multiple perspectives reduce errors
2. **Error Detection**: Critic identifies false positives/negatives
3. **Transparency**: Decision rationale provided for each classification
4. **Confidence Tracking**: Uncertainty flagged for human review
5. **Quality Assurance**: Built-in review process
6. **Consistency**: Arbitrator ensures uniform standards

## Trade-offs

### Advantages
- Higher classification accuracy
- Built-in quality control
- Detailed decision rationale
- Better handling of edge cases
- Reduced false positives/negatives

### Disadvantages
- 3x slower (three sequential API calls)
- 3x higher API costs
- More complex implementation
- Requires LangGraph dependency

## When to Use LangGraph Classifier

**Use LangGraph classifier when:**
- Accuracy is more important than speed
- You need decision rationale and transparency
- Working with complex or ambiguous contracts
- Quality assurance is critical
- Budget allows for higher API costs

**Use standard classifier when:**
- Speed is critical
- Processing large volumes
- Budget is constrained
- Contracts are straightforward
- Post-processing review is available

## Installation

### Requirements

```bash
pip install langgraph boto3 pyyaml
```

### Dependencies

- `langgraph`: For StateGraph workflow orchestration
- `boto3`: For AWS Bedrock API access
- `pyyaml`: For configuration file parsing
- Python 3.8+

## Output Format

The LangGraph classifier produces output compatible with the standard labeler format:

```python
{
    "contract_id": "contract_001",
    "status": "success",
    "error": None,
    "final_classification": {
        "Document Name": {
            "label": 1,
            "text": "MASTER SERVICES AGREEMENT",
            "decision_rationale": "Clearly stated in header"
        },
        "Parties": {
            "label": 1,
            "text": "Between Company A and Company B",
            "decision_rationale": "Explicitly identified in preamble"
        },
        # ... all 41 categories
    },
    "metadata": {
        "classifier_output": {...},
        "critic_output": {...},
        "arbitrator_summary": {...}
    }
}
```

## Evaluation

The LangGraph classifier can be evaluated using the same evaluation script:

```bash
python evaluate_labeler.py \
    --predictions labeled_contracts.csv \
    --ground-truth test.json \
    --output evaluation_results.json
```

## Future Enhancements

Potential improvements to the reflection pattern:

1. **Iterative Refinement**: Add loops for multiple reflection cycles
2. **Conditional Routing**: Skip critic for high-confidence classifications
3. **Parallel Processing**: Run multiple contracts in parallel
4. **Adaptive Confidence**: Adjust thresholds based on contract type
5. **Human-in-the-Loop**: Flag low-confidence decisions for human review
6. **Fine-tuning**: Train specialized models for each agent role

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Reflection Agents Blog Post](https://blog.langchain.com/reflection-agents/)
- [CUAD Dataset](https://www.atticusprojectai.org/cuad)
- [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)

## Support

For issues or questions:
1. Check the main CUAD processing README
2. Review LangGraph documentation
3. Examine the example code in `langgraph_classifier.py`
4. Test with a small sample contract first

## License

Same as the main CUAD repository.
