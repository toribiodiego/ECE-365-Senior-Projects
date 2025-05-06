import time
import torch
import numpy as np
import os
import argparse
import sys
from PIL import Image
from collections import OrderedDict
from sklearn.preprocessing import normalize

# Import the model loader and image processor from main.py
from main import ImageProcessor

# Add the current directory to sys.path to import our patched hubconf directly
from patched_hubconf import _resnet_face18

def load_model_custom(config, device):
    """Load a face recognition model based on configuration."""
    if config["model_type"] == "resnet18":
        # Create model
        model = _resnet_face18(
            use_se=config["use_se"],
            grayscale=config["grayscale"]
        )
        
        # Load pretrained weights if available
        if config.get("pretrained_path") and os.path.exists(config["pretrained_path"]):
            print(f"Loading pretrained weights from {config['pretrained_path']}")
            try:
                state_dict = torch.load(config["pretrained_path"], map_location=device)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # Remove "module." prefix if it exists (from DataParallel)
                    new_key = k.replace("module.", "")
                    # Handle changes in input channels for color/grayscale
                    if new_key == "conv1.weight" and v.size(1) == 1 and not config["grayscale"]:
                        # Expand grayscale weights to 3 channels
                        v = v.repeat(1, 3, 1, 1) / 3.0
                    new_state_dict[new_key] = v
                
                model.load_state_dict(new_state_dict)
                print("Successfully loaded pretrained weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Using randomly initialized weights instead")
        else:
            print("No pretrained weights found, using randomly initialized weights")
            
        model = model.to(device)
        model.eval()
        return model
    
    elif config["model_type"] == "edgeface":
        # Use torch.hub for EdgeFace
        model = torch.hub.load(
            repo_or_dir='otroshi/edgeface',
            model=config["edgeface_variant"],
            source='github',
            pretrained=True
        )
        
        if "q" in config.get("edgeface_variant", ""):
            model.to("cpu")
        else:
            model.to(device)
            
        model.eval()
        return model
    
    else:
        raise ValueError(f"Unsupported model type: {config['model_type']}")

def test_all_models(test_image_path, data_root, device):
    """Test all available models with the given test image."""
    # Relative path to the pretrained weights
    pretrained_path = "models/resnet18_110.pth"
    
    # Define all model configurations to test
    model_configs = [
        # ResNetFace variants
        {
            "model_type": "resnet18",
            "use_se": True,  # Now trying with SE=True
            "grayscale": True,
            "image_size": 128,
            "pretrained_path": pretrained_path,
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        },
        {
            "model_type": "resnet18",
            "use_se": False,
            "grayscale": True,
            "image_size": 128,
            "pretrained_path": pretrained_path,
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        },
        {
            "model_type": "resnet18",
            "use_se": True,  # Now trying with SE=True
            "grayscale": False,
            "image_size": 128,
            "pretrained_path": pretrained_path,
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        },
        {
            "model_type": "resnet18",
            "use_se": False,
            "grayscale": False,
            "image_size": 128,
            "pretrained_path": pretrained_path,
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        },
        # EdgeFace variants
        {
            "model_type": "edgeface",
            "edgeface_variant": "edgeface_xxs",
            "grayscale": False,
            "image_size": 128,
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        }
    ]

    print(f"\nTesting {len(model_configs)} model configurations...")
    
    results = {}
    for model_config in model_configs:
        try:
            print(f"\n{'='*80}")
            print(f"Testing model configuration: {model_config}")
            print(f"{'='*80}")
            
            # Create model with custom loader
            model = load_model_custom(model_config, device)
            
            # Create image processor
            img_processor = ImageProcessor(model_config)
            
            # Process test image
            print(f"Processing test image: {test_image_path}")
            test_img = img_processor.process_img(test_image_path)
            test_img = test_img.unsqueeze(0)  # Add batch dimension
            
            # Measure inference time
            start_time = time.time()
            processing_device = torch.device("cpu") if ("q" in model_config.get("edgeface_variant", "") and model_config["model_type"] == "edgeface") else device
            test_img = test_img.to(processing_device)
            
            # Run inference
            print(f"Running inference with {model_config['model_type']} model...")
            with torch.no_grad():
                output = model(test_img)
                if isinstance(output, dict) and "fea" in output:
                    embedding = output["fea"].squeeze().cpu().numpy()
                else:
                    embedding = output.squeeze().cpu().numpy()
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Get embedding details
            embedding_norm = np.linalg.norm(embedding)
            embedding = normalize(embedding.reshape(1, -1), axis=1, norm='l2').flatten()
            
            # Store results
            model_name = f"{model_config['model_type']}"
            if model_config['model_type'] == 'edgeface':
                model_name += f"_{model_config.get('edgeface_variant', '')}"
            else:
                model_name += f"_SE-{model_config.get('use_se', False)}_GS-{model_config.get('grayscale', False)}"
            
            results[model_name] = {
                'embedding_size': embedding.shape[0],
                'inference_time_ms': inference_time * 1000,
                'embedding_norm_before': embedding_norm,
                'device': processing_device.type,
                'embedding': embedding  # Store the actual embedding for comparison
            }
            
            print(f"Results for {model_name}:")
            print(f"  Embedding size: {embedding.shape[0]}")
            print(f"  Inference time: {inference_time * 1000:.2f} ms")
            print(f"  Embedding norm (before normalization): {embedding_norm:.4f}")
            print(f"  Device: {processing_device.type}")
            
        except Exception as e:
            print(f"Error testing model configuration {model_config}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # If we have multiple models, compare their embeddings
    if len(results) > 1:
        print("\nComparing embeddings between models...")
        model_names = list(results.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                emb1 = results[model1]['embedding']
                emb2 = results[model2]['embedding']
                
                # Compute cosine similarity between embeddings
                similarity = np.dot(emb1, emb2)
                print(f"Similarity between {model1} and {model2}: {similarity:.4f}")
    
    # Remove the embeddings from results before returning (they're large)
    for model_name in results:
        if 'embedding' in results[model_name]:
            del results[model_name]['embedding']
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test all face recognition models with a single image')
    parser.add_argument('--test_image', type=str, required=True, help='Path to the test image')
    parser.add_argument('--data_root', type=str, default='align/lfw-align-128', help='Root directory of dataset images')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test all models
    results = test_all_models(args.test_image, args.data_root, device)
    
    # Print summary of results
    print("\nSummary of Results:")
    print("=" * 80)
    print(f"{'Model':<30} | {'Embedding Size':<15} | {'Inference Time':<15} | {'Device':<10}")
    print("-" * 80)
    for model_name, result in results.items():
        print(f"{model_name:<30} | {result['embedding_size']:<15} | {result['inference_time_ms']:.2f} ms{' ':>8} | {result['device']:<10}")
    print("=" * 80) 