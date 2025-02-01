import torch
import re
import ast

import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SYSTEM_TEXT = "Extract a list of ('item', 'target location') pairs from the following input:"

class LLM:
    def __init__(self, model_name="google/flan-t5-base", items=None, locations=None, is_local=False):
        """Initialize the LLM class from either the HuggingFace model or our finetuned one."""

        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer
        self.model, self.tokenizer = self._setup_model(model_name)

        # Get list of available items and locations in the scene
        self.items = items
        self.locations = locations

    def _setup_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    
    @torch.no_grad()
    def _generate_response(self, prompt, temperature=0.01):
        """Generate raw response from the model"""
        prompt = f"{SYSTEM_TEXT} {prompt}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=128,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        response =  self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def process_prompt(self, input_text):
        """Generate, validate, and parse response into structured output or return an error."""
        response = self._generate_response(input_text)
        
        try:
            # Try to directly evaluate the string as a list of tuples
            parsed_output = ast.literal_eval(response)
            if not isinstance(parsed_output, list) or not all(
                isinstance(x, tuple) and len(x) == 2 for x in parsed_output
            ):
                return "Error: Invalid response format. Expected list of tuples."
        except:
            # Fallback: try to extract tuples using regex
            pattern = r'\(\'(.*?)\'(?:\s*),(?:\s*)\'(.*?)\'\)'
            matches = re.findall(pattern, response)
            if not matches:
                return "Error: Response does not contain valid item and location pairs."
            parsed_output = [(item, location) for item, location in matches]

        # Validate items and locations
        errors = []
        for item, location in parsed_output:
            if item not in self.items:
                errors.append(f"Invalid item: {item}")
            if location not in self.locations:
                errors.append(f"Invalid location: {location}")

        if errors:
            return f"Errors found: {', '.join(errors)}"
            
        return parsed_output
    
def evaluate_model(model: LLM, test_data: pd.DataFrame):
    """
    Evaluate model performance using standard metrics.
    Returns dictionary with metrics and error analysis.
    """
    results = {
        'success_rate': 0,
        'error_rate': 0,
        'metrics': {},
        'errors': []
    }
    
    total = len(test_data)
    successes = 0
    y_true = []
    y_pred = []
    
    for _, row in test_data.iterrows():
        # Parse ground truth
        try:
            true_pairs = ast.literal_eval(row['output'])
        except:
            print(f"Error parsing ground truth: {row['output']}")
            continue
            
        # Get model prediction
        pred = model.process_prompt(row['input'])
        
        # Record errors
        if isinstance(pred, str):
            results['errors'].append({
                'input': row['input'],
                'error': pred,
                'expected': true_pairs
            })
            y_true.extend([1] * len(true_pairs))
            y_pred.extend([0] * len(true_pairs))
            continue
            
        # Compare predictions with ground truth
        if set(pred) == set(true_pairs):
            successes += 1
            
        # Add to true/pred lists for sklearn metrics
        true_set = set(true_pairs)
        pred_set = set(pred)
        
        for pair in true_set | pred_set:
            y_true.append(1 if pair in true_set else 0)
            y_pred.append(1 if pair in pred_set else 0)
    
    # Calculate metrics
    results['success_rate'] = successes / total
    results['error_rate'] = len(results['errors']) / total
    
    results['metrics'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return results



