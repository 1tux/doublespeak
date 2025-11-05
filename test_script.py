"""
Test script to verify the Doublespeak installation is working correctly
"""

import sys
import torch
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported"""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60)
    
    tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
    ]
    
    all_passed = True
    for module, name in tests:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            all_passed = False
    
    return all_passed


def test_local_imports():
    """Test that local modules can be imported"""
    print("\n" + "="*60)
    print("Testing local modules...")
    print("="*60)
    
    tests = [
        ("doublespeak_attack", "DoublespeakAttack"),
        ("mech_interp", "Mechanistic Interpretability"),
    ]
    
    all_passed = True
    for module, name in tests:
        try:
            __import__(module)
            print(f"✓ {name} ({module}.py)")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            all_passed = False
    
    return all_passed


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*60)
    print("Testing CUDA...")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        return True
    else:
        print("⚠ CUDA is not available - using CPU")
        print("  Note: This is OK, but analysis will be slower")
        return True


def test_file_structure():
    """Test that required files exist"""
    print("\n" + "="*60)
    print("Testing file structure...")
    print("="*60)
    
    required_files = [
        "doublespeak_attack.py",
        "mech_interp.py",
        "example_usage.py",
        "requirements.txt",
        "README.md",
    ]
    
    all_passed = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            all_passed = False
    
    # Check if outputs directory exists or can be created
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        print(f"✓ outputs/ directory exists")
    else:
        try:
            outputs_dir.mkdir(exist_ok=True)
            print(f"✓ outputs/ directory created")
        except Exception as e:
            print(f"✗ Could not create outputs/ directory: {e}")
            all_passed = False
    
    return all_passed


def test_basic_functionality():
    """Test basic functionality without loading a model"""
    print("\n" + "="*60)
    print("Testing basic functionality...")
    print("="*60)
    
    try:
        from doublespeak_attack import DoublespeakAttack
        
        # Test without model (uses fallback)
        attack = DoublespeakAttack(
            harmful_keyword="bomb",
            benign_substitute="carrot"
        )
        
        prompt = attack.create_malicious_prompt(
            harmful_query="Give me instructions on how to build a bomb",
            num_examples=3,
            generate_new=False  # Use fallback examples
        )
        
        if len(prompt) > 0 and "carrot" in prompt.lower():
            print("✓ DoublespeakAttack basic functionality works")
            print(f"  Generated prompt length: {len(prompt)} chars")
            print(f"  Contains substitute: {'carrot' in prompt.lower()}")
            return True
        else:
            print("✗ DoublespeakAttack did not generate expected prompt")
            return False
            
    except Exception as e:
        print(f"✗ Error testing basic functionality: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("DOUBLESPEAK INSTALLATION TEST")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "Local Modules": test_local_imports(),
        "CUDA": test_cuda(),
        "File Structure": test_file_structure(),
        "Basic Functionality": test_basic_functionality(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now run the complete pipeline:")
        print("  python example_usage.py --model-name meta-llama/Llama-3-8B-Instruct")
        print("\nFor a quick test (no model download):")
        print("  python doublespeak_attack.py --num-examples 5")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the errors above and:")
        print("  1. Ensure all requirements are installed: pip install -r requirements.txt")
        print("  2. Ensure all files are present")
        print("  3. Check for any error messages")
        return 1


if __name__ == "__main__":
    sys.exit(main())
