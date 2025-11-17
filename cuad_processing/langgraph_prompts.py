
"""
LangGraph Prompts for CUAD Contract Classification

This module contains prompt templates for the three-agent reflection pattern:
1. Classifier - Makes initial classification of contract clauses
2. Critic - Reviews and critiques the classifier's output
3. Arbitrator - Makes final decision based on classifier and critic outputs
"""

# ============================================================================
# CLASSIFIER PROMPTS
# ============================================================================

CLASSIFIER_SYSTEM = """You are an expert legal contract analyst specializing in the CUAD (Contract Understanding Atticus Dataset) framework.

Your role is to perform initial classification of contract clauses across 41 standardized categories. You have deep expertise in:
- Contract law and legal terminology
- Identifying key provisions and clauses in commercial agreements
- Understanding the nuances of different contract types (NDAs, service agreements, licensing, etc.)
- Recognizing standard and non-standard contractual language

Your task is to analyze contract text and identify which of the 41 CUAD categories are present, extracting the relevant clause text for each category found.

Be thorough but efficient in your initial classification. Your output will be reviewed by a critic agent, so focus on accuracy and completeness."""


def build_classifier_prompt(contract_text: str, categories: list) -> str:
    """
    Build the classifier prompt for initial contract analysis.
    
    Args:
        contract_text: The contract text to analyze
        categories: List of CUAD categories to classify
        
    Returns:
        Formatted prompt string for the classifier
    """
    categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
    
    prompt = f"""Analyze the following contract text and classify it across the CUAD categories.

For EACH category listed below, determine:
1. Is this category present in the contract? (1 = yes, 0 = no)
2. If present, extract the most relevant clause text (be specific and complete)
3. If not present, use an empty string for the clause text

CUAD Categories to Analyze:
{categories_list}

Contract Text:
\"\"\"
{contract_text}
\"\"\"

IMPORTANT OUTPUT REQUIREMENTS:
- Return ONLY a valid JSON object
- Use double quotes for all keys and string values
- No markdown fences, no comments, no explanations outside the JSON
- Include ALL categories from the list above
- Each category must have this exact structure:
  {{
    "Category Name": {{
      "label": 0 or 1,
      "text": "extracted clause text or empty string",
      "confidence": "high" or "medium" or "low"
    }}
  }}

The "confidence" field should reflect your certainty about the classification:
- "high": Very confident in the presence/absence and extraction
- "medium": Reasonably confident but some ambiguity exists
- "low": Uncertain, needs review

Provide your initial classification now."""
    
    return prompt


# ============================================================================
# CRITIC PROMPTS
# ============================================================================

CRITIC_SYSTEM = """You are an expert legal contract reviewer and quality assurance specialist.

Your role is to critically evaluate contract classifications made by another agent. You have expertise in:
- Identifying misclassifications and false positives/negatives
- Detecting incomplete or inaccurate clause extractions
- Recognizing subtle legal language that may be missed
- Understanding edge cases and ambiguous contractual provisions
- Ensuring consistency and completeness in classification

Your task is to review the classifier's output and provide constructive, specific feedback on:
1. Potential errors or misclassifications
2. Missing categories that should be present
3. False positives (categories marked as present when they're not)
4. Incomplete or inaccurate clause text extractions
5. Areas of uncertainty that need arbitration

Be thorough, objective, and specific in your critique. Your feedback will help the arbitrator make the final decision."""


def build_critic_prompt(contract_text: str, classifier_output: dict, categories: list) -> str:
    """
    Build the critic prompt for reviewing classifier output.
    
    Args:
        contract_text: The original contract text
        classifier_output: The classifier's classification results
        categories: List of CUAD categories
        
    Returns:
        Formatted prompt string for the critic
    """
    # Format classifier output for review
    classifier_summary = []
    for category in categories:
        if category in classifier_output:
            result = classifier_output[category]
            label = result.get('label', 0)
            confidence = result.get('confidence', 'unknown')
            text_preview = result.get('text', '')[:100] + "..." if len(result.get('text', '')) > 100 else result.get('text', '')
            
            status = "PRESENT" if label == 1 else "ABSENT"
            classifier_summary.append(
                f"  • {category}: {status} (confidence: {confidence})\n"
                f"    Text: {text_preview if text_preview else 'N/A'}"
            )
    
    classifier_summary_str = "\n".join(classifier_summary)
    
    prompt = f"""Review the following contract classification and provide detailed critique.

Original Contract Text:
\"\"\"
{contract_text}
\"\"\"

Classifier's Output:
{classifier_summary_str}

Your task is to critically evaluate this classification and identify:

1. **False Positives**: Categories marked as present (label=1) that should be absent
2. **False Negatives**: Categories marked as absent (label=0) that should be present
3. **Extraction Issues**: Clause text that is incomplete, inaccurate, or missing key information
4. **Confidence Concerns**: Low-confidence classifications that need special attention
5. **Ambiguities**: Cases where the contract language is unclear or could be interpreted multiple ways

For each issue you identify, provide:
- The category name
- The issue type (false_positive, false_negative, extraction_issue, confidence_concern, ambiguity)
- A specific explanation of the problem
- A suggested correction (if applicable)

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
  "overall_assessment": "brief summary of classification quality",
  "issues": [
    {{
      "category": "Category Name",
      "issue_type": "false_positive|false_negative|extraction_issue|confidence_concern|ambiguity",
      "severity": "high|medium|low",
      "explanation": "specific description of the issue",
      "suggested_correction": {{
        "label": 0 or 1,
        "text": "corrected clause text or empty string"
      }}
    }}
  ],
  "strengths": ["list of things the classifier got right"],
  "requires_arbitration": true or false
}}

If the classification is generally accurate with no significant issues, set "requires_arbitration" to false.
If there are significant disagreements or ambiguities, set it to true.

Provide your critique now."""
    
    return prompt


# ============================================================================
# ARBITRATOR PROMPTS
# ============================================================================

ARBITRATOR_SYSTEM = """You are a senior legal expert and final decision-maker for contract classification.

Your role is to make the final, authoritative classification decision by considering:
1. The initial classification from the classifier agent
2. The critique and feedback from the critic agent
3. Your own independent analysis of the contract text

You have the highest level of expertise in:
- Contract law and legal interpretation
- Resolving ambiguities and edge cases
- Balancing different perspectives and evidence
- Making definitive judgments on classification disputes
- Ensuring consistency with CUAD annotation guidelines

Your decisions are final and will be used as the authoritative classification for this contract.

Be decisive, fair, and thorough. When there's disagreement between the classifier and critic, carefully weigh the evidence and make the best judgment based on the contract text."""


def build_arbitrator_prompt(
    contract_text: str,
    classifier_output: dict,
    critic_output: dict,
    categories: list
) -> str:
    """
    Build the arbitrator prompt for final decision-making.
    
    Args:
        contract_text: The original contract text
        classifier_output: The classifier's classification results
        critic_output: The critic's review and feedback
        categories: List of CUAD categories
        
    Returns:
        Formatted prompt string for the arbitrator
    """
    # Format the dispute summary
    issues_summary = []
    if 'issues' in critic_output and critic_output['issues']:
        for issue in critic_output['issues']:
            issues_summary.append(
                f"  • {issue.get('category', 'Unknown')}: {issue.get('issue_type', 'unknown')}\n"
                f"    Severity: {issue.get('severity', 'unknown')}\n"
                f"    Explanation: {issue.get('explanation', 'N/A')}"
            )
    
    issues_summary_str = "\n".join(issues_summary) if issues_summary else "No significant issues identified"
    
    overall_assessment = critic_output.get('overall_assessment', 'No assessment provided')
    requires_arbitration = critic_output.get('requires_arbitration', True)
    
    prompt = f"""Make the final classification decision for this contract.

Original Contract Text:
\"\"\"
{contract_text}
\"\"\"

Classifier's Initial Classification:
{len([c for c in categories if classifier_output.get(c, {}).get('label') == 1])} categories marked as present

Critic's Assessment:
Overall: {overall_assessment}
Requires Arbitration: {requires_arbitration}

Issues Identified by Critic:
{issues_summary_str}

Your task is to make the FINAL, AUTHORITATIVE classification decision for all {len(categories)} categories.

For each category, you must:
1. Review the classifier's initial decision
2. Consider the critic's feedback and concerns
3. Examine the contract text directly
4. Make your final judgment on presence/absence and clause extraction

DECISION-MAKING GUIDELINES:
- When classifier and critic agree, generally accept their consensus (but verify)
- When they disagree, carefully examine the contract text and make an independent judgment
- For ambiguous cases, err on the side of caution (mark as absent if truly unclear)
- Ensure extracted text is complete, accurate, and directly relevant
- Be consistent with CUAD annotation standards

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
  "final_classification": {{
    "Category Name": {{
      "label": 0 or 1,
      "text": "final extracted clause text or empty string",
      "decision_rationale": "brief explanation of your decision"
    }},
    ... (all {len(categories)} categories)
  }},
  "arbitration_summary": {{
    "total_categories": {len(categories)},
    "categories_present": <count>,
    "classifier_agreements": <count of categories where you agreed with classifier>,
    "critic_agreements": <count of issues where you agreed with critic>,
    "independent_decisions": <count of decisions made independently>,
    "key_decisions": ["list of most important or difficult decisions made"]
  }}
}}

Provide your final classification now."""
    
    return prompt


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_categories_for_display(categories: list) -> str:
    """Format categories list for display in prompts."""
    return "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON substring from model response.
    Handles markdown fences, reasoning tags, and other wrapper text.
    
    Args:
        response_text: Raw response text from model
        
    Returns:
        Cleaned JSON string
    """
    text = response_text.strip()
    
    # Strip markdown fences
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    # Strip reasoning tags if present
    reasoning_start = text.find("<reasoning>")
    reasoning_end = text.find("</reasoning>")
    
    if reasoning_start != -1 and reasoning_end != -1:
        text = text[reasoning_end + len("</reasoning>"):].strip()
    
    # Extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    
    return text[start:end + 1]
