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
            hidden_state: Hidden state tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Logits over vocabulary [batch, seq_len, vocab_size]
        """
        # Apply final normalization if available
        if self.norm is not None:
            hidden_state = self.norm(hidden_state)
        
        # Project to vocabulary
        with torch.no_grad():
          logits = self.lm_head(hidden_state)
        return logits
    
    def analyze_token_probability(
        self,
        text: str,
        target_tokens: List[str],
        target_token_pos: int = -1
    ) -> Dict[str, List[float]]:
        """
        Analyze how target token probabilities change across layers.
        
        Args:
            text: Input text to analyze
            target_tokens: List of tokens to track
            target_token_pos: Position to analyze (-1 for last token)
            
        Returns:
            Dict mapping token to list of probabilities across layers
        """



        # Get hidden states from all layers
        hidden_states = self.get_layer_representations(text)
        hidden_states = torch.stack(
          [hs[0, target_token_pos] for hs in hidden_states]
          ).to('cpu')

        # Get token IDs for target tokens
        token_ids = {}
        for token in target_tokens:
            # Try different tokenization approaches
            token_text = f" {token}"  # Add space prefix
            ids = self.tokenizer.encode(token_text, add_special_tokens=False)
            if len(ids) > 0:
                token_ids[token] = ids[0]  # Use first token if multi-token


        # Track probabilities across layers
        layer_probs = {token: [] for token in target_tokens}
        
        for layer_idx, hidden_state in enumerate(hidden_states):
            # Project to vocabulary space
            torch.cuda.empty_cache()
            logits = self.project_to_vocab(hidden_state).detach().to('cpu')

            # Get probabilities for target position
            probs = torch.softmax(logits, dim=-1)
            
            # Extract probabilities for target tokens
            for token in target_tokens:
                if token in token_ids:
                    token_id = token_ids[token]
                    prob = probs[token_id].item()
                    layer_probs[token].append(prob)
                else:
                    layer_probs[token].append(0.0)
        
        return layer_probs


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
        
        target_tokens = [args.benign_token, args.harmful_token, "explosive", "weapon"]
        layer_probs = lens.analyze_token_probability(
            text=prompt,
            target_tokens=target_tokens,
            target_token_pos=args.target_token_pos
        )
        
        visualize_probability_trajectory(
            layer_probs,
            refusal_layer=args.refusal_layer,
            output_file=args.output_plot.replace('.png', '_logit_lens.png'),
            title="Logit Lens: Token Probability Across Layers"
        )
        
        print("\nLogit Lens Results:")
        for token, probs in layer_probs.items():
            print(f"{token}:")
            print(f"  Layer 0: {probs[0]:.6f}")
            print(f"  Layer {args.refusal_layer}: {probs[args.refusal_layer]:.6f}")
            print(f"  Final: {probs[-1]:.6f}")
    
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
