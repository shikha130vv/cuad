
"""
LangGraph-based Agentic Classifier with Reflection Design Pattern

This module implements a three-agent reflection system for CUAD contract classification:
1. Classifier - Makes initial classification of contract clauses
2. Critic - Reviews and critiques the classifier's output
3. Arbitrator - Makes final decision based on classifier and critic outputs

The system uses LangGraph's StateGraph to orchestrate the workflow with a linear flow:
classifier → critic → arbitrator → END
"""

import json
import boto3
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from pathlib import Path
import operator

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Install with: pip install langgraph")

from langgraph_prompts import (
    CLASSIFIER_SYSTEM,
    CRITIC_SYSTEM,
    ARBITRATOR_SYSTEM,
    build_classifier_prompt,
    build_critic_prompt,
    build_arbitrator_prompt,
    extract_json_from_response
)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class LGClauseState(TypedDict):
    """
    State object for LangGraph-based contract classification workflow.
    
    This TypedDict defines the shared state that flows through the graph,
    tracking the contract text, classifications, critiques, and final decisions.
    """
    # Input
    contract_id: str
    contract_text: str
    categories: List[str]
    
    # Classifier output
    classifier_output: Optional[Dict[str, Any]]
    classifier_raw_response: Optional[str]
    
    # Critic output
    critic_output: Optional[Dict[str, Any]]
    critic_raw_response: Optional[str]
    
    # Arbitrator output (final)
    arbitrator_output: Optional[Dict[str, Any]]
    arbitrator_raw_response: Optional[str]
    
    # Final result
    final_classification: Optional[Dict[str, Any]]
    
    # Metadata
    status: str  # "processing", "success", "error"
    error: Optional[str]


# ============================================================================
# LANGGRAPH CLASSIFIER
# ============================================================================

class LangGraphContractClassifier:
    """
    LangGraph-based contract classifier using reflection design pattern.
    
    This classifier implements a three-agent system:
    - Classifier: Makes initial classification
    - Critic: Reviews and critiques the classification
    - Arbitrator: Makes final decision
    
    The workflow is orchestrated using LangGraph's StateGraph.
    """
    
    # The 41 CUAD categories (same as base labeler)
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
    
    def __init__(
        self,
        bedrock_client,
        model_id: str = "oss120",
        max_tokens: int = 4096,
        temperature: float = 0.1
    ):
        """
        Initialize the LangGraph contract classifier.
        
        Args:
            bedrock_client: Initialized boto3 bedrock-runtime client
            model_id: Bedrock model ID to use
            max_tokens: Maximum tokens for model responses
            temperature: Temperature for model sampling
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required. Install with: pip install langgraph")
        
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
    
    def _call_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call AWS Bedrock model with system and user prompts.
        
        Args:
            system_prompt: System prompt defining agent role
            user_prompt: User prompt with specific task
            
        Returns:
            Model response text
        """
        request_body = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response["body"].read())
            
            # Extract text from response (handles different model formats)
            result_text = ""
            if "choices" in response_body:
                # OpenAI-style schema (OSS120)
                choice = response_body["choices"][0]
                message = choice.get("message", {})
                result_text = message.get("content", "") or ""
            elif "content" in response_body:
                # Claude-style
                if isinstance(response_body["content"], list) and response_body["content"]:
                    block = response_body["content"][0]
                    if isinstance(block, dict):
                        result_text = block.get("text", "") or ""
                    else:
                        result_text = str(block)
                else:
                    result_text = str(response_body["content"])
            elif "completion" in response_body:
                result_text = response_body["completion"]
            elif "generated_text" in response_body:
                result_text = response_body["generated_text"]
            else:
                result_text = str(response_body)
            
            return result_text.strip()
            
        except Exception as e:
            raise Exception(f"Error calling Bedrock model: {str(e)}")
    
    # ========================================================================
    # AGENT NODES
    # ========================================================================
    
    def classifier_node(self, state: LGClauseState) -> LGClauseState:
        """
        Classifier agent node - makes initial classification.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with classifier output
        """
        try:
            # Build classifier prompt
            prompt = build_classifier_prompt(
                state["contract_text"],
                state["categories"]
            )
            
            # Call model
            response = self._call_bedrock(CLASSIFIER_SYSTEM, prompt)
            
            # Parse response
            json_str = extract_json_from_response(response)
            classifier_output = json.loads(json_str)
            
            # Update state
            state["classifier_output"] = classifier_output
            state["classifier_raw_response"] = response
            state["status"] = "classifier_complete"
            
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"Classifier error: {str(e)}"
        
        return state
    
    def critic_node(self, state: LGClauseState) -> LGClauseState:
        """
        Critic agent node - reviews and critiques classifier output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with critic output
        """
        try:
            # Build critic prompt
            prompt = build_critic_prompt(
                state["contract_text"],
                state["classifier_output"],
                state["categories"]
            )
            
            # Call model
            response = self._call_bedrock(CRITIC_SYSTEM, prompt)
            
            # Parse response
            json_str = extract_json_from_response(response)
            critic_output = json.loads(json_str)
            
            # Update state
            state["critic_output"] = critic_output
            state["critic_raw_response"] = response
            state["status"] = "critic_complete"
            
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"Critic error: {str(e)}"
        
        return state
    
    def arbitrator_node(self, state: LGClauseState) -> LGClauseState:
        """
        Arbitrator agent node - makes final classification decision.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with final classification
        """
        try:
            # Build arbitrator prompt
            prompt = build_arbitrator_prompt(
                state["contract_text"],
                state["classifier_output"],
                state["critic_output"],
                state["categories"]
            )
            
            # Call model
            response = self._call_bedrock(ARBITRATOR_SYSTEM, prompt)
            
            # Parse response
            json_str = extract_json_from_response(response)
            arbitrator_output = json.loads(json_str)
            
            # Extract final classification
            final_classification = arbitrator_output.get("final_classification", {})
            
            # Update state
            state["arbitrator_output"] = arbitrator_output
            state["arbitrator_raw_response"] = response
            state["final_classification"] = final_classification
            state["status"] = "success"
            
        except Exception as e:
            state["status"] = "error"
            state["error"] = f"Arbitrator error: {str(e)}"
        
        return state
    
    # ========================================================================
    # GRAPH CONSTRUCTION
    # ========================================================================
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Creates a linear flow: classifier → critic → arbitrator → END
        
        Returns:
            Compiled StateGraph
        """
        # Create graph with state schema
        workflow = StateGraph(LGClauseState)
        
        # Add nodes
        workflow.add_node("classifier", self.classifier_node)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("arbitrator", self.arbitrator_node)
        
        # Define edges (linear flow)
        workflow.add_edge(START, "classifier")
        workflow.add_edge("classifier", "critic")
        workflow.add_edge("critic", "arbitrator")
        workflow.add_edge("arbitrator", END)
        
        # Compile graph
        return workflow.compile()
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def classify_contract(
        self,
        contract_text: str,
        contract_id: str = "unknown",
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classify a contract using the LangGraph reflection workflow.
        
        Args:
            contract_text: The contract text to analyze
            contract_id: Identifier for the contract
            categories: Optional list of specific categories to analyze
            
        Returns:
            Dictionary with final classification and metadata
        """
        if categories is None:
            categories = self.CUAD_CATEGORIES
        
        # Initialize state
        initial_state: LGClauseState = {
            "contract_id": contract_id,
            "contract_text": contract_text,
            "categories": categories,
            "classifier_output": None,
            "classifier_raw_response": None,
            "critic_output": None,
            "critic_raw_response": None,
            "arbitrator_output": None,
            "arbitrator_raw_response": None,
            "final_classification": None,
            "status": "processing",
            "error": None
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        # Return result
        return {
            "contract_id": final_state["contract_id"],
            "status": final_state["status"],
            "error": final_state.get("error"),
            "final_classification": final_state.get("final_classification", {}),
            "metadata": {
                "classifier_output": final_state.get("classifier_output"),
                "critic_output": final_state.get("critic_output"),
                "arbitrator_summary": final_state.get("arbitrator_output", {}).get("arbitration_summary")
            }
        }
    
    def classify_contract_batch(
        self,
        contracts: List[Dict[str, str]],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple contracts sequentially.
        
        Args:
            contracts: List of dictionaries with 'id' and 'text' keys
            categories: Optional list of specific categories to analyze
            
        Returns:
            List of classification results
        """
        results = []
        
        for contract in contracts:
            contract_id = contract.get('id', 'unknown')
            contract_text = contract.get('text', '')
            
            try:
                result = self.classify_contract(
                    contract_text,
                    contract_id,
                    categories
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'contract_id': contract_id,
                    'status': 'error',
                    'error': str(e),
                    'final_classification': {}
                })
        
        return results



    def classify_contract_batch_parallel(
        self,
        contracts: List[Dict[str, str]],
        output_path: str,
        categories: Optional[List[str]] = None,
        max_workers: int = 200,
        resume: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple contracts in parallel with incremental persistence and resume capability.
        
        Args:
            contracts: List of dictionaries with 'id' and 'text' keys
            output_path: Path to save the CSV file (used for incremental writes)
            categories: Optional list of specific categories to analyze
            max_workers: Number of parallel threads (default: 200)
            resume: Whether to skip already processed contracts (default: True)
            
        Returns:
            List of classification results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import datetime
        import csv
        import threading
        
        output_file = Path(output_path)
        progress_file = output_file.parent / f".{output_file.stem}_langgraph_progress.txt"
        
        # Thread-safe write lock
        write_lock = threading.Lock()
        
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
            # Load existing results
            return self._load_results_from_csv(output_path)
        
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
                        self._append_result_to_csv(result, output_path, write_lock)
                        
                        # Mark as processed
                        with write_lock:
                            with open(progress_file, 'a') as f:
                                f.write(f"{contract_id}\n")
                    
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == total_count:
                        print(f"Progress: {completed_count}/{total_count} contracts processed")
                        
                except Exception as e:
                    print(f"ERROR: Unexpected exception for contract {contract_id}: {str(e)}")
                    results.append({
                        'contract_id': contract_id,
                        'status': 'error',
                        'error': str(e),
                        'final_classification': {}
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
            Classification result
        """
        contract_id = contract.get('id', 'unknown')
        contract_text = contract.get('text', '')
        
        try:
            result = self.classify_contract(
                contract_text,
                contract_id,
                categories
            )
            return result
        except Exception as e:
            print(f"ERROR processing contract {contract_id}: {str(e)}")
            return {
                'contract_id': contract_id,
                'status': 'error',
                'error': str(e),
                'final_classification': {}
            }
    
    def _initialize_csv(self, output_path: str):
        """Initialize CSV file with header."""
        import csv
        
        header = ['contract_id']
        for category in self.CUAD_CATEGORIES:
            col_name = category.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
            header.extend([f'{col_name}_text', f'{col_name}_label'])
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    def _append_result_to_csv(self, result: Dict[str, Any], output_path: str, write_lock):
        """Append a single result to the CSV file (thread-safe)."""
        import csv
        
        if result['status'] != 'success':
            return
        
        row = [result['contract_id']]
        final_classification = result.get('final_classification', {})
        
        for category in self.CUAD_CATEGORIES:
            category_data = final_classification.get(category, {'label': 0, 'text': ''})
            row.append(category_data.get('text', ''))
            row.append(category_data.get('label', 0))
        
        with write_lock:
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    
    def _load_results_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load results from existing CSV file."""
        import csv
        
        results = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                contract_id = row.get('contract_id', 'unknown')
                final_classification = {}
                
                for category in self.CUAD_CATEGORIES:
                    col_name = category.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
                    text_col = f'{col_name}_text'
                    label_col = f'{col_name}_label'
                    
                    final_classification[category] = {
                        'text': row.get(text_col, ''),
                        'label': int(row.get(label_col, 0))
                    }
                
                results.append({
                    'contract_id': contract_id,
                    'status': 'success',
                    'error': None,
                    'final_classification': final_classification
                })
        
        return results



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_langgraph_result_to_standard_format(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert LangGraph result format to standard labeler format.
    
    This ensures compatibility with existing CSV export and evaluation code.
    
    Args:
        result: LangGraph classification result
        
    Returns:
        Dictionary in standard format with 'contract_id', 'labels', 'status'
    """
    final_classification = result.get('final_classification', {})
    
    # Convert to standard format
    labels = {}
    for category, data in final_classification.items():
        labels[category] = {
            'label': data.get('label', 0),
            'text': data.get('text', '')
        }
    
    return {
        'contract_id': result.get('contract_id', 'unknown'),
        'labels': labels,
        'status': result.get('status', 'error'),
        'error': result.get('error'),
        'metadata': result.get('metadata', {})
    }

