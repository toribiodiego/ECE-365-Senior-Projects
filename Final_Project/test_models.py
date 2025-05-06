#!/usr/bin/env python3
"""
Test script to verify access to EdgeFace models.
This script attempts to load each EdgeFace variant using our local patched_hubconf.py.
"""

import os
import sys
import torch
import time

# Try to import our modules
try:
    from main import model_dict as EDGEFACE_VARIANTS
except ImportError:
    # Default EdgeFace variants if import fails
    EDGEFACE_VARIANTS = [
        'base', 's_gamma_05', 'xs_q', 'xs_gamma_06', 'xxs', 'xxs_q'
    ]
    print("Warning: Could not import model_dict from main.py, using default variants.")

def test_edgeface_models():
    """Try to load each EdgeFace model variant using our local patched_hubconf.py."""
    print("=" * 60)
    print("Testing EdgeFace model variants using local patched_hubconf.py")
    print("=" * 60)
    
    # First, make sure hubconf.py is present (copy from patched_hubconf.py if needed)
    if not os.path.exists("hubconf.py") and os.path.exists("patched_hubconf.py"):
        print("Copying patched_hubconf.py to hubconf.py...")
        import shutil
        shutil.copy("patched_hubconf.py", "hubconf.py")
    
    successful_models = []
    failed_models = []
    
    for variant in EDGEFACE_VARIANTS:
        model_name = f"edgeface_{variant}" if not variant.startswith('edgeface_') else variant
        # Extract variant without 'edgeface_' prefix
        variant_name = variant.replace('edgeface_', '')
        print(f"\nTesting {model_name}...")
        
        # Determine if model should use CPU (quantized models must use CPU)
        use_cpu = "q" in variant
        device = torch.device("cpu") if use_cpu or not torch.cuda.is_available() else "cuda"
        print(f"Using device: {device}")
        
        try:
            start_time = time.time()
            # Try to load the model using our local patched_hubconf.py
            model = torch.hub.load(
                repo_or_dir='.',  # Look in current directory
                model=variant_name,  # Model name in patched_hubconf.py
                source='local',  # Use local hubconf
                pretrained=True
            )
            
            # Move to appropriate device
            model.to(device)
            model.eval()
            
            # Basic test with random input
            input_shape = (1, 3, 128, 128)  # Batch, Channels, Height, Width
            dummy_input = torch.randn(input_shape).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output shape
            if isinstance(output, torch.Tensor):
                output_shape = output.shape
                print(f"✅ Model loaded successfully!")
                print(f"   Input shape: {input_shape}")
                print(f"   Output shape: {output_shape}")
                print(f"   Load time: {time.time() - start_time:.2f} seconds")
                successful_models.append(model_name)
            else:
                print(f"❌ Model output has unexpected format: {type(output)}")
                failed_models.append(model_name)
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            failed_models.append(model_name)
    
    # Summary
    print("\n" + "=" * 60)
    if successful_models:
        print(f"✅ Successfully loaded {len(successful_models)}/{len(EDGEFACE_VARIANTS)} models:")
        for model in successful_models:
            print(f"   - {model}")
    
    if failed_models:
        print(f"❌ Failed to load {len(failed_models)}/{len(EDGEFACE_VARIANTS)} models:")
        for model in failed_models:
            print(f"   - {model}")
    
    print("=" * 60)
    
    return len(successful_models) > 0

if __name__ == "__main__":
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Run the test
        success = test_edgeface_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 