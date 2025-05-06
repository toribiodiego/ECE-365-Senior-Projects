import os
import argparse
import subprocess
import time

def main():
    # Get the script's directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description="Generate embeddings for all models")
    parser.add_argument("--data_root", type=str, 
                       default=os.path.join(project_root, "align/lfw-align-128"), 
                       help="Root directory of the face dataset")
    parser.add_argument("--test_image", type=str, 
                       default=os.path.join(project_root, "test_images/arnold.jpeg"),
                       help="Test image to use for embedding generation")
    args = parser.parse_args()
    
    # Verify the test image exists
    if not os.path.exists(args.test_image):
        print(f"Error: Test image {args.test_image} not found.")
        return
    
    # Verify the data directory exists
    if not os.path.exists(args.data_root):
        print(f"Error: Data directory {args.data_root} not found.")
        return
    
    # List of models to generate embeddings for
    models = [
        "edgeface_xxs",  # Fastest model, do this first
        "resnet18_gs",   # Standard ResNet grayscale
        "resnet18_color", # Color variant
        "resnet18_se_gs", # With SE blocks
        "resnet18_se_color" # SE with color
    ]
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(script_dir, "embeddings_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    total_start_time = time.time()
    
    # Run the find_similar_enhanced.py script for each model
    for i, model in enumerate(models):
        print(f"\n{'='*80}")
        print(f"Generating embeddings for model ({i+1}/{len(models)}): {model}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Use the full path to the script
        script_path = os.path.join(script_dir, "find_similar_enhanced.py")
        
        cmd = [
            "python", script_path, 
            args.test_image, args.test_image,
            "--model", model,
            "--data_root", args.data_root
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            end_time = time.time()
            print(f"\nCompleted embedding generation for {model}")
            print(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")
        except subprocess.CalledProcessError as e:
            print(f"\nError generating embeddings for {model}: {e}")
            continue
    
    total_end_time = time.time()
    total_minutes = (total_end_time - total_start_time) / 60
    
    print(f"\n{'='*80}")
    print(f"All embedding generation complete!")
    print(f"Total time: {total_minutes:.2f} minutes ({total_minutes/60:.2f} hours)")
    print(f"{'='*80}")
    
    # List cache files
    cache_files = os.listdir(cache_dir)
    print("\nGenerated cache files:")
    for file in cache_files:
        file_path = os.path.join(cache_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  {file} ({file_size:.2f} MB)")

if __name__ == "__main__":
    main() 