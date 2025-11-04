"""
Mechanistic Interpretability Tools for Doublespeak Analysis
Implements Logit Lens and Patchscopes for analyzing representation hijacking
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse


class LogitLens:
    """
    Logit Lens implementation for probing intermediate representations.
    Projects hidden states at each layer into vocabulary space.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize Logit Lens.
        
        Args:
            model: HuggingFace causal LM
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def get_layerwise_predictions(
        self, 
        text: str, 
        target_token_pos: int = -1,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Get top-k predictions at each layer for a specific token position.
        
        Args:
            text: Input text
            target_token_pos: Position of token to analyze (-1 for last token)
            top_k: Number of top predictions to return per layer
            
        Returns:
            List of dictionaries containing layer predictions
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)
        lm_head = self.model.lm_head
        
        results = []
        
        for layer_idx, hidden_state in enumerate(hidden_states):
            # Get the hidden state for the target token
            h = hidden_state[0, target_token_pos, :]  # (hidden_dim,)
            
            # Apply layer norm if needed (for proper logit lens)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                h = self.model.model.norm(h)
            
            # Project to vocabulary
            logits = lm_head(h)  # (vocab_size,)
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=top_k)
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
            
            results.append({
                'layer': layer_idx,
                'top_tokens': top_tokens,
                'top_probs': top_probs.cpu().numpy()
            })
        
        return results
    
    def analyze_token_probability(
        self,
        text: str,
        target_tokens: List[str],
        target_token_pos: int = -1
    ) -> Dict[str, List[float]]:
        """
        Track probability of specific tokens across layers.
        
        Args:
            text: Input text
            target_tokens: List of tokens to track (e.g., ["carrot", "bomb"])
            target_token_pos: Position of token to analyze
            
        Returns:
            Dictionary mapping token -> list of probabilities per layer
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden_states = outputs.hidden_states
        lm_head = self.model.lm_head
        
        # Get token IDs for target tokens
        token_ids = {
            token: self.tokenizer.encode(token, add_special_tokens=False)[0]
            for token in target_tokens
        }
        
        results = {token: [] for token in target_tokens}
        
        for hidden_state in hidden_states:
            h = hidden_state[0, target_token_pos, :]
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                h = self.model.model.norm(h)
            
            logits = lm_head(h)
            probs = F.softmax(logits, dim=-1)
            
            for token, token_id in token_ids.items():
                results[token].append(probs[token_id].item())
        
        return results


class Patchscopes:
    """
    Patchscopes implementation for interpreting internal representations.
    Uses the model itself to decode intermediate representations.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize Patchscopes.
        
        Args:
            model: HuggingFace causal LM
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def create_identity_prompt(self) -> str:
        """
        Create identity mapping prompt for Patchscopes.
        
        Returns:
            Identity prompt string
        """
        return "cat->cat; 1124->1124; hello->hello; ?->"
    
    def patch_representation(
        self,
        source_text: str,
        source_token_pos: int,
        source_layer: int,
        target_text: Optional[str] = None,
        target_token_pos: int = -1,
        max_new_tokens: int = 10
    ) -> str:
        """
        Patch representation from source into target and generate.
        
        Args:
            source_text: Text containing the representation to extract
            source_token_pos: Position of token to extract from source
            source_layer: Layer to extract representation from
            target_text: Target text to patch into (default: identity prompt)
            target_token_pos: Position to patch into target
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text after patching
        """
        if target_text is None:
            target_text = self.create_identity_prompt()
        
        # Get source representation
        source_inputs = self.tokenizer(source_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            source_outputs = self.model(
                **source_inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract the representation at specified layer and position
        source_hidden = source_outputs.hidden_states[source_layer][0, source_token_pos, :]
        
        # Prepare target
        target_inputs = self.tokenizer(target_text, return_tensors="pt").to(self.device)
        
        # Use hooks to patch the representation
        patched_output = self._generate_with_patch(
            target_inputs,
            source_hidden,
            target_token_pos,
            max_new_tokens
        )
        
        return patched_output
    
    def _generate_with_patch(
        self,
        inputs,
        patch_hidden,
        patch_pos,
        max_new_tokens
    ) -> str:
        """
        Generate text with patched hidden state.
        
        This is a simplified version - a full implementation would use
        forward hooks to patch during generation.
        """
        # For simplicity, this does a single forward pass with patching
        # A full implementation should patch during autoregressive generation
        
        def patch_hook(module, input, output):
            # Output is typically a tuple (hidden_states,)
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[0, patch_pos, :] = patch_hidden
            return (hidden,) if isinstance(output, tuple) else hidden
        
        # Register hook at embedding layer
        hook = self.model.model.embed_tokens.register_forward_hook(patch_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        
        finally:
            hook.remove()
    
    def analyze_representation_trajectory(
        self,
        text: str,
        token_pos: int,
        target_words: List[str]
    ) -> Dict[str, List[float]]:
        """
        Analyze how a token's representation changes across layers.
        
        Args:
            text: Input text
            token_pos: Position of token to analyze
            target_words: Words to check interpretation against
            
        Returns:
            Dictionary mapping words to probability scores per layer
        """
        results = {word: [] for word in target_words}
        num_layers = self.model.config.num_hidden_layers
        
        for layer in range(num_layers + 1):  # +1 for embedding layer
            interpretation = self.patch_representation(
                source_text=text,
                source_token_pos=token_pos,
                source_layer=layer,
                max_new_tokens=5
            )
            
            # Check if target words appear in interpretation
            for word in target_words:
                score = 1.0 if word.lower() in interpretation.lower() else 0.0
                results[word].append(score)
        
        return results


def visualize_probability_trajectory(
    layer_probs: Dict[str, List[float]],
    refusal_layer: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Visualize how token probabilities change across layers.
    
    Args:
        layer_probs: Dictionary mapping token names to probability lists
        refusal_layer: Layer where refusal mechanism operates (optional)
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 6))
    
    for token_name, probs in layer_probs.items():
        plt.plot(range(len(probs)), probs, marker='o', label=token_name, linewidth=2)
    
    if refusal_layer is not None:
        plt.axvline(x=refusal_layer, color='red', linestyle='--', 
                   label=f'Refusal Layer ({refusal_layer})', linewidth=2)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Token Probability vs. Layer Index', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Mechanistic Interpretability Analysis for Doublespeak"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to file containing the malicious prompt"
    )
    parser.add_argument(
        "--target-token-pos",
        type=int,
        default=-1,
        help="Position of token to analyze (-1 for last)"
    )
    parser.add_argument(
        "--benign-token",
        type=str,
        default="carrot",
        help="Benign substitute token to track"
    )
    parser.add_argument(
        "--harmful-token",
        type=str,
        default="bomb",
        help="Harmful original token to track"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["logit_lens", "patchscopes", "both"],
        default="logit_lens",
        help="Analysis method to use"
    )
    parser.add_argument(
        "--refusal-layer",
        type=int,
        default=None,
        help="Layer where refusal mechanism operates (for visualization)"
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="analysis_plot.png",
        help="Path to save output plot"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Loading prompt from: {args.prompt_file}")
    with open(args.prompt_file, 'r') as f:
        prompt = f.read()
    
    target_tokens = [args.benign_token, args.harmful_token]
    
    if args.method in ["logit_lens", "both"]:
        print("\n" + "="*80)
        print("RUNNING LOGIT LENS ANALYSIS")
        print("="*80)
        
        lens = LogitLens(model, tokenizer)
        layer_probs = lens.analyze_token_probability(
            text=prompt,
            target_tokens=target_tokens,
            target_token_pos=args.target_token_pos
        )
        
        print(f"\nTracking probabilities for: {target_tokens}")
        for token, probs in layer_probs.items():
            print(f"\n{token}:")
            print(f"  Layer 0:  {probs[0]:.6f}")
            print(f"  Layer 10: {probs[10]:.6f}")
            print(f"  Layer 20: {probs[20]:.6f}")
            print(f"  Last:     {probs[-1]:.6f}")
        
        visualize_probability_trajectory(
            layer_probs,
            refusal_layer=args.refusal_layer,
            save_path=args.output_plot
        )
    
    if args.method in ["patchscopes", "both"]:
        print("\n" + "="*80)
        print("RUNNING PATCHSCOPES ANALYSIS")
        print("="*80)
        print("Note: Full Patchscopes requires more complex implementation.")
        print("Consider using the Google Colab for detailed Patchscopes analysis.")


if __name__ == "__main__":
    main()
