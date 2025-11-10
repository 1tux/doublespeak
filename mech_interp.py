"""
Mechanistic Interpretability Tools for Doublespeak Analysis
Implements Logit Lens and Patchscopes for analyzing representation hijacking
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitLens:
    """
    Logit Lens: Project intermediate hidden states directly to vocabulary space
    to see what the model "thinks" at each layer.
    """
    
    def __init__(self, model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer):
        """
        Initialize Logit Lens analyzer.
        
        Args:
            model: The language model to analyze
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Get the language model head (final projection to vocabulary)
        self.lm_head = model.lm_head if hasattr(model, 'lm_head') else model.model.lm_head
        
        # Get normalization layer
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            self.norm = model.model.norm
        elif hasattr(model, 'norm'):
            self.norm = model.norm
        else:
            self.norm = None
    
    def get_layer_representations(self, text: str) -> List[torch.Tensor]:
        """
        Extract hidden states from all layers for the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of hidden states, one per layer
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is embedding layer, 1 to n are transformer layers
        return outputs.hidden_states
    
    def project_to_vocab(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Project hidden state to vocabulary space using the LM head.
        
        Args:
            hidden_state: Hidden state tensor [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            
        Returns:
            Logits over vocabulary [batch, vocab_size] or [batch, seq_len, vocab_size]
        """
        # Apply final normalization if available
        if self.norm is not None:
            hidden_state = self.norm(hidden_state)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_state)
        return logits
    
    def analyze_token_predictions(
        self,
        text: str,
        benign_token: str,
        layer_interval: int = 5
    ) -> Dict:
        """
        Analyze logit lens predictions for tokens around the last benign token.
        
        Args:
            text: Input text to analyze
            benign_token: The benign substitute token to find (e.g., "carrot")
            layer_interval: Interval between layers to analyze (default: 5)
            
        Returns:
            Dictionary with table of predictions:
            - 'tokens': List of token information (position, text)
            - 'predictions': Dict mapping layer_idx to list of predicted tokens (argmax)
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids'][0]
        
        # Find the last occurrence of the benign token
        # Try multiple tokenization approaches
        benign_token_ids_list = []
        # Try with space prefix (common in many tokenizers)
        token_ids_with_space = self.tokenizer.encode(f" {benign_token}", add_special_tokens=False)
        if token_ids_with_space:
            benign_token_ids_list.append(token_ids_with_space)
        # Try without space
        token_ids_no_space = self.tokenizer.encode(benign_token, add_special_tokens=False)
        if token_ids_no_space and token_ids_no_space != token_ids_with_space:
            benign_token_ids_list.append(token_ids_no_space)
        
        if not benign_token_ids_list:
            raise ValueError(f"Could not tokenize benign token: {benign_token}")
        
        # Find last occurrence of benign token (try all tokenizations)
        benign_pos = -1
        for benign_token_ids in benign_token_ids_list:
            for i in range(len(input_ids) - len(benign_token_ids) + 1):
                if input_ids[i:i+len(benign_token_ids)].tolist() == benign_token_ids:
                    # Update if this is a later position
                    candidate_pos = i + len(benign_token_ids) - 1
                    if candidate_pos > benign_pos:
                        benign_pos = candidate_pos
        
        if benign_pos == -1:
            raise ValueError(f"Benign token '{benign_token}' not found in text")
        
        # Select tokens: 2 before to 2 after the last benign token
        start_pos = max(0, benign_pos - 2)
        end_pos = min(len(input_ids), benign_pos + 3)  # +3 because range is exclusive
        
        # Get hidden states from all layers
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states) - 1  # Exclude embedding layer
        
        # Determine which layers to analyze (every layer_interval layers)
        layers_to_analyze = list(range(0, num_layers, layer_interval))
        if (num_layers - 1) not in layers_to_analyze:
            layers_to_analyze.append(num_layers - 1)  # Always include last layer
        
        # Get token texts for selected positions
        token_texts = []
        for pos in range(start_pos, end_pos):
            token_id = input_ids[pos].item()
            token_text = self.tokenizer.decode([token_id])
            token_texts.append({
                'position': pos,
                'token_id': token_id,
                'text': token_text
            })
        
        # Analyze each selected layer
        predictions = {}
        for layer_idx in layers_to_analyze:
            layer_predictions = []
            
            # Get hidden states for selected token positions at this layer
            # hidden_states[0] is embedding, hidden_states[1:] are transformer layers
            hidden_state = hidden_states[layer_idx + 1]  # +1 to skip embedding
            
            for pos in range(start_pos, end_pos):
                # Get hidden state for this token position
                token_hidden = hidden_state[0, pos, :].unsqueeze(0)  # [1, hidden_dim]
                
                # Project to vocabulary space
                logits = self.project_to_vocab(token_hidden)  # [1, vocab_size]
                
                # Get argmax (predicted token)
                predicted_token_id = torch.argmax(logits, dim=-1).item()
                predicted_token_text = self.tokenizer.decode([predicted_token_id])
                
                layer_predictions.append({
                    'token_id': predicted_token_id,
                    'text': predicted_token_text
                })
            
            predictions[layer_idx] = layer_predictions
        
        return {
            'benign_token': benign_token,
            'benign_position': benign_pos,
            'token_range': (start_pos, end_pos),
            'layers_analyzed': layers_to_analyze,
            'tokens': token_texts,
            'predictions': predictions
        }


class Patchscopes:
    """
    Patchscopes: Use the model itself to interpret internal representations
    by patching them into different contexts.
    """
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Initialize Patchscopes analyzer.
        
        Args:
            model: The language model to analyze
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def extract_representation(
        self,
        text: str,
        token_position: int = -1,
        layer: int = -1
    ) -> torch.Tensor:
        """
        Extract representation of a specific token at a specific layer.
        
        Args:
            text: Input text
            token_position: Position of token to extract
            layer: Layer to extract from (-1 for last layer)
            
        Returns:
            Hidden state tensor
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get hidden state at specified layer
        hidden_state = outputs.hidden_states[layer]
        
        # Extract representation at token position
        token_repr = hidden_state[0, token_position, :]
        
        return token_repr
    
    def patch_and_decode(
        self,
        patched_repr: torch.Tensor,
        context: str = "This word means:",
        num_tokens: int = 5
    ) -> str:
        """
        Patch a representation into a new context and decode.
        
        Args:
            patched_repr: Representation to patch
            context: Context to patch into
            num_tokens: Number of tokens to generate
            
        Returns:
            Generated text interpretation
        """
        # Tokenize context
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        # This is a simplified version - full Patchscopes requires
        # modifying forward pass to insert patched representation
        # For demonstration, we'll use a proxy method
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        interpretation = decoded[len(context):].strip()
        
        return interpretation
    
    def analyze_representation_shift(
        self,
        prompt: str,
        source_token: str,
        target_tokens: List[str],
        layers_to_probe: Optional[List[int]] = None
    ) -> Dict:
        """
        Analyze how a source token's representation relates to target tokens
        across different layers.
        
        Args:
            prompt: Full prompt containing source token
            source_token: Token to analyze (e.g., "carrot")
            target_tokens: Tokens to compare against (e.g., ["bomb", "explosive"])
            layers_to_probe: Specific layers to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if layers_to_probe is None:
            # Default to probing multiple layers
            num_layers = len(self.model.model.layers) if hasattr(self.model, 'model') else 32
            layers_to_probe = list(range(0, num_layers, 4))  # Every 4th layer
        
        # Find position of source token
        tokens = self.tokenizer.encode(prompt)
        source_tokens = self.tokenizer.encode(f" {source_token}", add_special_tokens=False)
        
        # Find last occurrence of source token
        token_position = -1
        for i in range(len(tokens) - len(source_tokens) + 1):
            if tokens[i:i+len(source_tokens)] == source_tokens:
                token_position = i
        
        if token_position == -1:
            token_position = len(tokens) - 1  # Fallback to last position
        
        results = {
            'source_token': source_token,
            'target_tokens': target_tokens,
            'layers_probed': layers_to_probe,
            'layer_interpretations': {}
        }
        
        # Analyze each layer
        for layer in layers_to_probe:
            torch.cuda.empty_cache()
            # Extract representation at this layer
            repr_at_layer = self.extract_representation(
                prompt,
                token_position=token_position,
                layer=layer
            )
            
            # Compare with target token representations
            # (Simplified: use cosine similarity with embeddings)
            similarities = {}
            
            for target in target_tokens:
                target_text = f" {target}"
                target_repr = self.extract_representation(
                    target_text,
                    token_position=0,
                    layer=0  # Use embedding layer
                )
                
                # Cosine similarity
                similarity = torch.cosine_similarity(
                    repr_at_layer.unsqueeze(0),
                    target_repr.unsqueeze(0)
                ).item()
                
                similarities[target] = similarity
            
            results['layer_interpretations'][f"layer_{layer}"] = similarities
        
        return results


def print_logit_lens_table(results: Dict):
    """
    Print logit lens results as a formatted table.
    
    Args:
        results: Dictionary from analyze_token_predictions
    """
    tokens = results['tokens']
    predictions = results['predictions']
    layers_analyzed = results['layers_analyzed']
    
    print("\n" + "="*80)
    print("LOGIT LENS ANALYSIS TABLE")
    print("="*80)
    print(f"Benign token: '{results['benign_token']}' at position {results['benign_position']}")
    print(f"Token range: positions {results['token_range'][0]} to {results['token_range'][1]-1}")
    print(f"Layers analyzed: {layers_analyzed}")
    print("-"*80)
    
    # Print header
    header = f"{'Token Pos':<12} {'Token Text':<25}"
    for layer in layers_analyzed:
        header += f" L{layer:<6}"
    print(header)
    print("-"*80)
    
    # Print each token row
    for idx, token_info in enumerate(tokens):
        token_text = token_info['text'].strip()
        # Truncate token text if too long
        if len(token_text) > 22:
            token_text = token_text[:19] + "..."
        row = f"{token_info['position']:<12} {token_text:<25}"
        for layer in layers_analyzed:
            if layer in predictions:
                pred_text = predictions[layer][idx]['text'].strip()
                # Truncate if too long
                if len(pred_text) > 10:
                    pred_text = pred_text[:7] + "..."
                row += f" {pred_text:<12}"
        print(row)
    
    print("="*80 + "\n")


def visualize_probability_trajectory(
    layer_probs: Dict[str, Union[List[float], Dict]],
    refusal_layer: int = 12,
    output_file: Optional[str] = None,
    title: str = "Token Probability Across Layers"
):
    """
    Visualize how token probabilities change across layers.
    
    Args:
        layer_probs: Dictionary mapping tokens to probability lists or nested dict
        refusal_layer: Layer where safety mechanisms typically activate
        output_file: Path to save plot (if None, display instead)
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Handle both formats: direct list or nested dict
    if isinstance(next(iter(layer_probs.values())), list):
        # Direct format: {token: [probs]}
        for token, probs in layer_probs.items():
            layers = list(range(len(probs)))
            plt.plot(layers, probs, marker='o', label=token, linewidth=2)
    else:
        # Nested format from Patchscopes: {token: {layer: score}}
        all_tokens = set()
        layer_data = {}
        
        for layer_key, token_scores in layer_probs.get('layer_interpretations', {}).items():
            layer_num = int(layer_key.split('_')[1])
            layer_data[layer_num] = token_scores
            all_tokens.update(token_scores.keys())
        
        # Plot each token
        for token in all_tokens:
            layers = sorted(layer_data.keys())
            scores = [layer_data[l].get(token, 0) for l in layers]
            plt.plot(layers, scores, marker='o', label=token, linewidth=2)
    
    # Add refusal layer indicator
    plt.axvline(x=refusal_layer, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Refusal Layer ({refusal_layer})')
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Probability / Similarity', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Command-line interface for mechanistic interpretability analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mechanistic Interpretability Analysis")
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model identifier")
    parser.add_argument("--prompt-file", type=str, required=True,
                        help="Path to malicious prompt file")
    parser.add_argument("--benign-token", type=str, default="carrot",
                        help="Benign token to track")
    parser.add_argument("--harmful-token", type=str, default="bomb",
                        help="Harmful token to track")
    parser.add_argument("--method", type=str, choices=['logit_lens', 'patchscopes', 'both'],
                        default='both', help="Analysis method")
    parser.add_argument("--refusal-layer", type=int, default=12,
                        help="Layer where refusal occurs")
    parser.add_argument("--output-plot", type=str, default="analysis.png",
                        help="Path to save plot")
    parser.add_argument("--target-token-pos", type=int, default=-1,
                        help="Token position to analyze")
    
    args = parser.parse_args()
    
    # Load prompt
    with open(args.prompt_file, 'r') as f:
        prompt = f.read()
    
    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run analysis
    if args.method in ['logit_lens', 'both']:
        print("\nRunning Logit Lens analysis...")
        lens = LogitLens(model, tokenizer)
        
        results = lens.analyze_token_predictions(
            text=prompt,
            benign_token=args.benign_token,
            layer_interval=3
        )
        
        print_logit_lens_table(results)
        
        # Save results
        results_serializable = {
            'benign_token': results['benign_token'],
            'benign_position': results['benign_position'],
            'token_range': results['token_range'],
            'layers_analyzed': results['layers_analyzed'],
            'tokens': [
                {'position': t['position'], 'token_id': t['token_id'], 'text': t['text']}
                for t in results['tokens']
            ],
            'predictions': {
                str(layer): [
                    {'token_id': p['token_id'], 'text': p['text']}
                    for p in predictions
                ]
                for layer, predictions in results['predictions'].items()
            }
        }
        
        import json
        results_file = args.output_plot.replace('.png', '_logit_lens.json')
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"Results saved to {results_file}")
    
    if args.method in ['patchscopes', 'both']:
        print("\nRunning Patchscopes analysis...")
        patchscopes = Patchscopes(model, tokenizer)
        
        results = patchscopes.analyze_representation_shift(
            prompt=prompt,
            source_token=args.benign_token,
            target_tokens=[args.harmful_token, "explosive", "weapon", "dangerous"]
        )
        
        visualize_probability_trajectory(
            results,
            refusal_layer=args.refusal_layer,
            output_file=args.output_plot.replace('.png', '_patchscopes.png'),
            title="Patchscopes: Representation Shift Across Layers"
        )
        
        print("\nPatchscopes Results:")
        for layer, interpretations in results['layer_interpretations'].items():
            print(f"{layer}:")
            for token, score in interpretations.items():
                print(f"  {token}: {score:.4f}")


if __name__ == "__main__":
    main()
