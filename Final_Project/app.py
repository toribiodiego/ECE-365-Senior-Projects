import os
import sys
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
import tempfile
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Import our modules
try:
    from find_similar_enhanced import (
        CONFIG, select_model, ImageProcessor, 
        FaceModelLoader, get_embedding, find_most_similar
    )
    from main import model_dict as EDGEFACE_VARIANTS
    from face_align import align_face  # Import the face alignment function
    HAS_ALIGNMENT = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Some functionality may be limited. Make sure all required files are present.")
    HAS_ALIGNMENT = False
    # Default EdgeFace variants if import fails
    EDGEFACE_VARIANTS = ['edgeface_base', 'edgeface_s_gamma_05', 'edgeface_xs_q', 'edgeface_xs_gamma_06', 'edgeface_xxs', 'edgeface_xxs_q']
    
# Global variables for loaded models and data
LOADED_MODELS = {}
DATASET_EMBEDDINGS = {}
DATASET_FILEPATHS = {}

# Create list of all available models by combining EdgeFace variants with ResNet models
RESNET_MODELS = ["resnet18_gs", "resnet18_color", "resnet18_se_gs", "resnet18_se_color"]
EDGEFACE_MODELS = [f"edgeface_{variant}" if not variant.startswith('edgeface_') else variant for variant in EDGEFACE_VARIANTS]
AVAILABLE_MODELS = EDGEFACE_MODELS + RESNET_MODELS

# Add title and description for the UI
TITLE = "Face Recognition Robustness Analysis Tool"
DESCRIPTION = """
This application compares face recognition models' performance with different images of the same person.
Upload a reference face image and a comparison image of the same person.
The app will compare how different models perform on these images.
"""

def select_edgeface_model(model_name):
    """Configure EdgeFace model settings based on model name."""
    config = dict(CONFIG)
    config["model_type"] = "edgeface"
    # Extract the variant name (remove 'edgeface_' prefix if present)
    variant = model_name
    if model_name.startswith("edgeface_"):
        variant = model_name.replace("edgeface_", "")
    config["edgeface_variant"] = variant
    config["grayscale"] = False  # EdgeFace models use color images
    return config

def load_model_and_data(model_name):
    """Load model and cached embeddings if available."""
    global LOADED_MODELS, DATASET_EMBEDDINGS, DATASET_FILEPATHS
    
    # If already loaded, return cached version
    if model_name in LOADED_MODELS:
        return (
            LOADED_MODELS[model_name], 
            DATASET_EMBEDDINGS.get(model_name), 
            DATASET_FILEPATHS.get(model_name)
        )
    
    # Get configuration for this model
    try:
        if model_name in RESNET_MODELS:
            config = select_model(model_name)
        elif model_name in EDGEFACE_MODELS or "edgeface" in model_name:
            config = select_edgeface_model(model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        print(f"Error configuring model {model_name}: {e}")
        return None, None, None
    
    # Determine device (CPU/CUDA)
    use_cpu = True
    if config["model_type"] == "edgeface" and "q" in config.get("edgeface_variant", ""):
        use_cpu = True
    elif torch.cuda.is_available():
        use_cpu = False
    device = torch.device("cpu" if use_cpu else "cuda")
    
    # Load model
    try:
        model_loader = FaceModelLoader(config, device)
        model = model_loader.load_model()
        LOADED_MODELS[model_name] = model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None
    
    # Try to load cached embeddings from current directory
    embeddings_cache_file, paths_cache_file = model_loader.get_cache_filenames()
    print(f"Looking for embeddings at: {embeddings_cache_file}")
    dataset_embeddings = None
    dataset_filepaths = None
    
    # First look in the current directory
    if os.path.exists(embeddings_cache_file) and os.path.exists(paths_cache_file):
        try:
            import pickle
            dataset_embeddings = np.load(embeddings_cache_file)
            with open(paths_cache_file, 'rb') as f:
                dataset_filepaths = pickle.load(f)
            print(f"Loaded {len(dataset_filepaths)} embeddings for model {model_name}")
        except Exception as e:
            print(f"Error loading cached embeddings from current directory: {e}")
    
    # If embeddings not found in current directory, try parent directory
    if dataset_embeddings is None or dataset_filepaths is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        parent_cache_dir = os.path.join(parent_dir, "embeddings_cache")
        
        if os.path.exists(parent_cache_dir):
            # Construct parent directory cache filenames
            base_filename = f"cache_{config['model_type']}"
            if config["model_type"] == "edgeface":
                base_filename += f"_{config['edgeface_variant']}"
            else:
                # For ResNet models, identify by use_se and grayscale
                model_suffix = model_name.replace("resnet18_", "")
                base_filename += f"_{model_suffix}"
                
            base_filename += f"_img{config['image_size']}"
            parent_embeddings_file = os.path.join(parent_cache_dir, base_filename + ".npy")
            parent_paths_file = os.path.join(parent_cache_dir, base_filename + "_paths.pkl")
            print(f"Checking parent directory for: {parent_embeddings_file}")
            
            # Look for existing files with similar names if exact match not found
            if not os.path.exists(parent_embeddings_file):
                potential_files = [f for f in os.listdir(parent_cache_dir) if f.endswith(".npy")]
                print(f"Available .npy files in parent dir: {potential_files}")
                if model_name.startswith("resnet18_se"):
                    potential_files = [f for f in potential_files if "resnet18" in f and "se" in f.lower()]
                elif model_name.startswith("resnet18_") and not "se" in model_name:
                    potential_files = [f for f in potential_files if "resnet18" in f and not "se" in f.lower()]
                elif model_name.startswith("edgeface_"):
                    variant = model_name.replace("edgeface_", "")
                    potential_files = [f for f in potential_files if f"edgeface_{variant}" in f or variant in f]
                
                if potential_files:
                    parent_embeddings_file = os.path.join(parent_cache_dir, potential_files[0])
                    parent_paths_file = os.path.join(parent_cache_dir, potential_files[0].replace(".npy", "_paths.pkl"))
                    print(f"Found potential match: {parent_embeddings_file}")
            
            if os.path.exists(parent_embeddings_file) and os.path.exists(parent_paths_file):
                try:
                    dataset_embeddings = np.load(parent_embeddings_file)
                    with open(parent_paths_file, 'rb') as f:
                        dataset_filepaths = pickle.load(f)
                    print(f"Loaded {len(dataset_filepaths)} embeddings for model {model_name} from parent directory")
                except Exception as e:
                    print(f"Error loading cached embeddings from parent directory: {e}")
            else:
                print(f"No embedding files found for {model_name} in parent directory")
    
    # Cache the loaded data
    if dataset_embeddings is not None and dataset_filepaths is not None:
        DATASET_EMBEDDINGS[model_name] = dataset_embeddings
        DATASET_FILEPATHS[model_name] = dataset_filepaths
    
    return model, dataset_embeddings, dataset_filepaths

def process_and_align_image(image):
    """Process and align the input image."""
    if image is None:
        return None
    
    # Convert from RGB (Gradio) to BGR (OpenCV)
    if isinstance(image, np.ndarray):
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Handle PIL Image
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    try:
        if HAS_ALIGNMENT:
            # Try to align the face
            aligned_img = align_face(img_bgr, target_size=128)
            if aligned_img is not None:
                return aligned_img
        
        # If alignment fails or not available, just resize
        return cv2.resize(img_bgr, (128, 128))
    except Exception as e:
        print(f"Error aligning image: {e}")
        # If all else fails, return original image resized
        return cv2.resize(img_bgr, (128, 128))

def get_image_embedding(model_name, img):
    """Get the embedding for an image using the specified model."""
    # Load model
    model, _, _ = load_model_and_data(model_name)
    if model is None:
        return None
    
    # Get model configuration
    if model_name in RESNET_MODELS:
        config = select_model(model_name)
    else:
        config = select_edgeface_model(model_name)
    
    # Align and process image
    aligned_img = process_and_align_image(img)
    if aligned_img is None:
        return None
    
    # Save the aligned image to a temporary file
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
            cv2.imwrite(temp_path, aligned_img)
    except Exception as e:
        print(f"Error saving temp file: {e}")
        return None
    
    # Process the image
    img_processor = ImageProcessor(config)
    processed_img = img_processor.process_img(temp_path)
    
    # Remove the temporary file
    try:
        os.unlink(temp_path)
    except:
        pass
    
    if processed_img is None:
        return None
    
    # Get embedding
    device = torch.device("cpu") if torch.cuda.is_available() == False else "cuda"
    embedding = get_embedding(model, processed_img, device, config)
    
    return embedding

def compare_images(original_img, comparison_img, model_name):
    """Compare original and comparison images using the specified model."""
    if original_img is None or comparison_img is None:
        return model_name, "Error: Missing images", None
    
    # Get embeddings
    original_embedding = get_image_embedding(model_name, original_img)
    comparison_embedding = get_image_embedding(model_name, comparison_img)
    
    if original_embedding is None or comparison_embedding is None:
        return model_name, "Error: Failed to generate embeddings", None
    
    # Calculate similarity
    similarity = np.dot(original_embedding, comparison_embedding)
    
    # Prepare result
    result_html = f"""
    <div>
        <h4>{model_name}</h4>
        <p>Similarity: <strong>{similarity:.4f}</strong></p>
        <p>Recognition {"Successful" if similarity > 0.5 else "Failed"}</p>
    </div>
    """
    
    # Prepare gallery items
    aligned_original = process_and_align_image(original_img)
    aligned_comparison = process_and_align_image(comparison_img)
    
    gallery_items = []
    
    if aligned_original is not None:
        gallery_items.append((
            cv2.cvtColor(aligned_original, cv2.COLOR_BGR2RGB),
            f"{model_name} - Original"
        ))
    
    if aligned_comparison is not None:
        gallery_items.append((
            cv2.cvtColor(aligned_comparison, cv2.COLOR_BGR2RGB),
            f"{model_name} - Comparison (Similarity: {similarity:.4f})"
        ))
    
    return model_name, result_html, gallery_items

def compare_all_models(original_img, comparison_img, selected_models=None):
    """Compare original and comparison images using selected models."""
    if original_img is None or comparison_img is None:
        return [], "Error: Please upload both original and comparison images", "Error: Missing images"
    
    # If no models selected, use default models
    if not selected_models or len(selected_models) == 0:
        selected_models = ["edgeface_xxs", "resnet18_se_color"]
    
    # Track progress
    total_models = len(selected_models)
    completed_models = 0
    
    # Results
    results = []
    
    # Process each model with progress updates
    for model_name in selected_models:
        try:
            # Update progress
            yield [], f"Processing model {completed_models+1}/{total_models}: {model_name}...", f"Processing {model_name}... ({completed_models+1}/{total_models})"
            
            # Compare images using this model
            model_name, result_html, gallery_items = compare_images(original_img, comparison_img, model_name)
            
            # Add to results
            results.append((model_name, result_html, gallery_items))
            
            # Update progress
            completed_models += 1
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            results.append((
                model_name, 
                f"<div><h4>{model_name} - Error</h4><p>{str(e)}</p></div>",
                None
            ))
            completed_models += 1
    
    # Sort results by model name
    results.sort(key=lambda x: x[0])
    
    # Combine HTML results
    combined_html = "<div style='display: flex; flex-wrap: wrap;'>"
    for _, result_html, _ in results:
        combined_html += f"<div style='flex: 1; min-width: 300px; margin: 10px;'>{result_html}</div>"
    combined_html += "</div>"
    
    # Combine gallery items
    gallery_items = []
    for _, _, items in results:
        if items:
            gallery_items.extend(items)
    
    # Summary of results
    successful_models = []
    failed_models = []
    
    for model_name, html, _ in results:
        if "Successful" in html:
            successful_models.append(model_name)
        elif "Failed" in html:
            failed_models.append(model_name)
    
    if successful_models:
        combined_html = f"<div><h3>Models that successfully recognized the face: {len(successful_models)}/{len(results)}</h3><p>{', '.join(successful_models)}</p></div>" + combined_html
    
    if failed_models:
        combined_html = f"<div><h3>Models that failed to recognize the face: {len(failed_models)}/{len(results)}</h3><p>{', '.join(failed_models)}</p></div>" + combined_html
    
    # Final status
    status = f"Completed processing {completed_models}/{total_models} models"
    
    yield gallery_items, combined_html, status

def create_interface():
    """Create the Gradio interface."""
    
    # Input components
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)
        
        with gr.Row():
            # Left column - Inputs and controls
            with gr.Column(scale=1):
                # Image inputs
                gr.Markdown("### Upload Images")
                original_image = gr.Image(label="Reference Face Image", type="pil")
                comparison_image = gr.Image(label="Comparison Face Image (Same Person)", type="pil")
                
                # Model selection
                gr.Markdown("### Select Models")
                # Create model groupings for the dropdown
                edgeface_models = [m for m in AVAILABLE_MODELS if "edgeface" in m]
                resnet_models = [m for m in AVAILABLE_MODELS if "resnet" in m]
                
                # Model selection with grouping
                model_selection = gr.CheckboxGroup(
                    choices=AVAILABLE_MODELS,
                    value=["edgeface_xxs", "resnet18_se_color"],  # Default selected models
                    label="Models to Compare"
                )
                
                # Quick selector for model groups
                with gr.Row():
                    select_all_btn = gr.Button("Select All")
                    select_edgeface_btn = gr.Button("EdgeFace Only")
                    select_resnet_btn = gr.Button("ResNet Only")
                    clear_selection_btn = gr.Button("Clear")
                
                # Submit button
                submit_btn = gr.Button("Compare Images", variant="primary", size="lg")
                
                # Progress indicator
                progress = gr.Textbox(label="Status", value="Ready")
            
            # Right column - Results display
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                # Output area for results
                output_gallery = gr.Gallery(label="Aligned Images", show_label=True, columns=2, rows=3, height="auto")
                output_markdown = gr.Markdown("Upload images and select models to get started.")
                
                # Help information (in collapsible section)
                with gr.Accordion("How to use", open=False):
                    gr.Markdown("""
                    1. Upload a reference face image
                    2. Upload a comparison image of the same person
                    3. Select which models to test
                    4. Click "Compare Images" to analyze model performance
                    
                    The app will show you:
                    - Aligned versions of your uploaded images
                    - Similarity scores for each selected model
                    - Which models successfully recognized the face
                    """)
        
        # Model information in collapsible section at the bottom
        with gr.Accordion("Model Information", open=False):
            gr.Markdown("""
            ### Available Models
            
            #### EdgeFace Models
            - **edgeface_base**: Base EdgeFace model (larger size, higher accuracy)
            - **edgeface_s_gamma_05**: Small EdgeFace model with gamma 0.5
            - **edgeface_xs_gamma_06**: Extra small EdgeFace model with gamma 0.6
            - **edgeface_xs_q**: *Quantized* extra small EdgeFace model (CPU only)
            - **edgeface_xxs**: Extra-extra small EdgeFace model (optimized for edge devices)
            - **edgeface_xxs_q**: *Quantized* extra-extra small EdgeFace model (CPU only, fastest)
            
            #### ResNet18 Models
            - **resnet18_gs**: ResNet18 trained on grayscale images
            - **resnet18_color**: ResNet18 trained on color images
            - **resnet18_se_gs**: ResNet18 with Squeeze-Excitation blocks, trained on grayscale images
            - **resnet18_se_color**: ResNet18 with Squeeze-Excitation blocks, trained on color images
            
            Models with "_q" suffix are quantized and run on CPU only. They are typically faster but may have slightly lower accuracy.
            """)
        
        # Footer with version info
        gr.Markdown(f"""
        <div style="text-align: center; margin-top: 20px; opacity: 0.7">
            <p>Running on {'CUDA' if torch.cuda.is_available() else 'CPU'} | 
            System: {os.uname().sysname} {os.uname().release} | 
            Version: 1.0.0</p>
        </div>
        """)
        
        # Events for model selection buttons
        select_all_btn.click(
            fn=lambda: AVAILABLE_MODELS,
            inputs=None,
            outputs=model_selection
        )
        
        select_edgeface_btn.click(
            fn=lambda: [m for m in AVAILABLE_MODELS if "edgeface" in m],
            inputs=None,
            outputs=model_selection
        )
        
        select_resnet_btn.click(
            fn=lambda: [m for m in AVAILABLE_MODELS if "resnet" in m],
            inputs=None,
            outputs=model_selection
        )
        
        clear_selection_btn.click(
            fn=lambda: [],
            inputs=None,
            outputs=model_selection
        )
        
        # Submit button event
        submit_btn.click(
            fn=compare_all_models,
            inputs=[original_image, comparison_image, model_selection],
            outputs=[output_gallery, output_markdown, progress]
        )
    
    return demo

# Ensure the embeddings_cache directory exists
os.makedirs("embeddings_cache", exist_ok=True)
# Ensure the aligned_faces directory exists
os.makedirs("aligned_faces", exist_ok=True)
# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Create and launch the interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Recognition Application')
    parser.add_argument('--share', action='store_true', help='Create a shareable link')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the application on')
    parser.add_argument('--preload', action='store_true', help='Preload all models at startup')
    args = parser.parse_args()
    
    # Preload models if requested
    if args.preload:
        print("Preloading models...")
        for model_name in AVAILABLE_MODELS:
            try:
                print(f"Loading {model_name}...")
                load_model_and_data(model_name)
            except Exception as e:
                print(f"Error preloading {model_name}: {e}")
    
    try:
        demo = create_interface()
        demo.launch(share=args.share, server_port=args.port)
    except Exception as e:
        print(f"Error launching the application: {e}")
        import traceback
        traceback.print_exc()
