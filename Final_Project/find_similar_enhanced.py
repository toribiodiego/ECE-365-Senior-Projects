import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from PIL import Image
from collections import OrderedDict
from sklearn.preprocessing import normalize
import glob # For finding image files
import pickle # For saving/loading the file paths list
import argparse # For command-line arguments
import sys
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Try to import optional dependencies
try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Import visualization utilities if available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# The base configuration (will be updated based on model selection)
CONFIG = {
    "pretrained_path": "models/resnet18_110.pth",  # Relative path to pretrained weights
    "model_type": "resnet18",               # Options: "resnet18" or "edgeface"
    "edgeface_variant": "edgeface_xxs",     # Used when model_type is "edgeface"
    "use_se": False,                        # Whether to use Squeeze-Excitation blocks (ResNetFace)
    "grayscale": True,                      # For ResNetFace
    "image_size": 128,                      # Input image size (width, height)
    "embedding_size": 512,                  # Expected embedding dimension from the model
    "cache_dir": "embeddings_cache",        # Directory to store cached embeddings
}

####################################
# PATCHED MODEL DEFINITION
####################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)
    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        return out

class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True, grayscale=True, embedding_size=512):
        super(ResNetFace, self).__init__()
        self.inplanes = 64
        in_ch = 1 if grayscale else 3
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], use_se)
        self.layer2 = self._make_layer(block, 128, layers[1], use_se, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], use_se, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], use_se, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, embedding_size)
        self.bn5 = nn.BatchNorm1d(embedding_size)

        # Patched Initialization - Fixed to handle modules with no bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, use_se, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)
        return {'fea': x}

def resnet_face18(use_se=True, grayscale=True, embedding_size=512):
    """Helper to create a ResNetFace-18 model."""
    return ResNetFace(IRBlock, [2, 2, 2, 2],
                    use_se=use_se,
                    grayscale=grayscale,
                    embedding_size=embedding_size)

####################################
# ENHANCED MODEL LOADER CLASS
####################################
class FaceModelLoader:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Create a descriptive model name for caching purposes
        if config["model_type"] == "resnet18":
            self.model_name = f"resnet18_SE-{config['use_se']}_GS-{config['grayscale']}"
        else:
            self.model_name = f"edgeface_{config['edgeface_variant']}"
        print(f"Initializing model: {self.model_name}")

    def load_model(self):
        if self.config["model_type"] == "resnet18":
            # Create model directly
            model = resnet_face18(
                use_se=self.config["use_se"],
                grayscale=self.config["grayscale"],
                embedding_size=self.config["embedding_size"]
            )
            
            # Load pretrained weights if available
            if self.config.get("pretrained_path") and os.path.exists(self.config["pretrained_path"]):
                print(f"Loading pretrained weights from {self.config['pretrained_path']}")
                try:
                    state_dict = torch.load(self.config["pretrained_path"], map_location=self.device)
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_key = k.replace("module.", "")
                        # If the model is configured for 3-channel input but the checkpoint has 1-channel conv1 weights, replicate them
                        if new_key == "conv1.weight":
                            if (not self.config["grayscale"]) and (v.size(1) == 1):
                                v = v.repeat(1, 3, 1, 1) / 3.0
                        new_state_dict[new_key] = v
                    
                    # Try strict loading first, fall back to non-strict for SE models
                    try:
                        model.load_state_dict(new_state_dict, strict=True)
                        print("Successfully loaded pretrained weights (strict)")
                    except Exception as e:
                        if self.config["use_se"]:
                            print(f"Strict loading failed: {e}")
                            print("Trying non-strict loading for SE model")
                            model.load_state_dict(new_state_dict, strict=False)
                            print("Successfully loaded pretrained weights (non-strict)")
                        else:
                            raise e
                except Exception as e:
                    print(f"Error loading pretrained weights: {e}")
                    print("Using randomly initialized weights instead")
            else:
                print("No pretrained weights found, using randomly initialized weights")
                
            model = model.to(self.device)
            model.eval()
            return model

        elif self.config["model_type"] == "edgeface":
            try:
                # Get the variant name without 'edgeface_' prefix
                variant = self.config["edgeface_variant"]
                if variant.startswith("edgeface_"):
                    variant = variant.replace("edgeface_", "")
                
                print(f"Loading EdgeFace model variant: {variant}")
                # Load from local hubconf.py (not from GitHub)
                model = torch.hub.load(
                    repo_or_dir='.', 
                    model=variant,
                    source='local', 
                    pretrained=True
                )
                
                if "q" in self.config["edgeface_variant"]:
                    model.to("cpu")
                    print(f"Loaded EdgeFace model '{self.config['edgeface_variant']}' (Quantized - CPU).")
                else:
                    model.to(self.device)
                    print(f"Loaded EdgeFace model '{self.config['edgeface_variant']}' (Device: {self.device}).")
                model.eval()
                return model
            except Exception as e:
                raise RuntimeError(f"Failed to load EdgeFace model '{self.config['edgeface_variant']}': {e}")
        else:
            raise ValueError("Unsupported model type: {}".format(self.config["model_type"]))

    def get_cache_filenames(self):
        """Generates filenames for cached embeddings based on config."""
        os.makedirs(self.config["cache_dir"], exist_ok=True)
        base_filename = f"cache_{self.model_name}_img{self.config['image_size']}"
        embeddings_file = os.path.join(self.config["cache_dir"], base_filename + ".npy")
        paths_file = os.path.join(self.config["cache_dir"], base_filename + "_paths.pkl")
        return embeddings_file, paths_file

####################################
# IMAGE PROCESSOR CLASS
####################################
class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.image_size = self.config["image_size"]
        # Determine grayscale based on model type AND config flag
        if self.config["model_type"] == "edgeface":
            self.is_grayscale = False  # EdgeFace models typically expect color
        elif self.config["model_type"] == "resnet18":
            self.is_grayscale = self.config.get("grayscale", True)  # Use config flag for ResNet
        else:
            self.is_grayscale = False  # Default to color
        print(f"ImageProcessor initialized. Grayscale: {self.is_grayscale}")

    def process_img(self, img_input):
        """
        Processes an image for the model.

        Args:
            img_input (str or np.ndarray): Path to the image file or an image
                                          loaded as an OpenCV BGR numpy array.

        Returns:
            torch.Tensor or None: Processed image tensor ready for the model,
                                  or None if processing fails.
        """
        img = None
        try:
            # --- Load image if path is provided ---
            if isinstance(img_input, str):
                img_path = img_input
                # Load according to grayscale setting needed by the model
                if self.is_grayscale:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR format
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # -> RGB format
                if img is None:
                    print(f"Warning: Could not read image file: {img_path}")
                    return None
            # --- Use provided NumPy array ---
            elif isinstance(img_input, np.ndarray):
                img_arr = img_input
                # Assume input array is BGR (common from OpenCV)
                # Convert to RGB or Grayscale as needed by the model
                if self.is_grayscale:
                    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:  # BGR to Gray
                        img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                    elif len(img_arr.shape) == 2:  # Already Gray
                        img = img_arr
                    else:  # Unexpected format
                        print(f"Warning: Cannot convert input array shape {img_arr.shape} to grayscale.")
                        return None
                else:  # Need RGB
                    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:  # BGR to RGB
                        img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                    elif len(img_arr.shape) == 2:  # Gray to RGB
                        img = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
                    else:  # Unexpected format
                        print(f"Warning: Cannot convert input array shape {img_arr.shape} to RGB.")
                        return None
            else:
                print(f"Error: Invalid input type for process_img: {type(img_input)}")
                return None

            # --- Common Processing Steps ---
            # Resize
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

            # Add channel dimension if grayscale
            if self.is_grayscale and len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)  # Shape becomes (H, W, 1)

            # Ensure 3 channels if color model expects it (e.g., handling grayscale images for color model)
            if not self.is_grayscale and len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Shape becomes (H, W, 3)

            # Transpose and Normalize HWC -> CHW and scale to [-1, 1]
            img = img.transpose((2, 0, 1)).astype(np.float32)
            img = (img - 127.5) / 127.5

            return torch.from_numpy(img).float()

        except Exception as e:
            # Include input type in error message for clarity
            input_desc = f"path '{img_input}'" if isinstance(img_input, str) else f"array shape {getattr(img_input, 'shape', 'N/A')}"
            print(f"Error processing image ({input_desc}): {e}")
            return None

####################################
# HELPER FUNCTIONS FOR INFERENCE & CACHING
####################################

def get_embedding(model, processed_img_tensor, device, config):
    """Gets embedding for a single processed image tensor."""
    if processed_img_tensor is None: 
        return None
    # Use CPU if model requires it (quantized)
    use_cpu = ("q" in config.get("edgeface_variant", "") and config["model_type"] == "edgeface")
    processing_device = torch.device("cpu") if use_cpu else device
    
    input_tensor = processed_img_tensor.unsqueeze(0).to(processing_device)
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict) and "fea" in output: 
            embedding = output["fea"].squeeze().cpu().numpy()
        elif torch.is_tensor(output): 
            embedding = output.squeeze().cpu().numpy()
        else: 
            raise TypeError(f"Unexpected model output type: {type(output)}")
        
        # Normalize the embedding here to ensure consistency
        embedding = normalize(embedding.reshape(1, -1), axis=1, norm='l2').flatten()
    return embedding

def get_all_dataset_embeddings(model, img_processor, data_root, device, config):
    """Generates embeddings for all valid images found in the data_root directory."""
    print(f"Generating dataset embeddings FROM SCRATCH: {data_root}")
    all_embeddings, all_filepaths = [], []
    start_time = time.time()
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(data_root, "**", ext), recursive=True))
    print(f"Found {len(image_paths)} potential image files.")
    processed_count, error_count = 0, 0
    for img_path in image_paths:
        processed_img = img_processor.process_img(img_path)
        if processed_img is not None:
            embedding = get_embedding(model, processed_img, device, config)
            if embedding is not None:
                all_embeddings.append(embedding)
                all_filepaths.append(img_path)
                processed_count += 1
            else: 
                error_count += 1
        else: 
            error_count += 1
        if (processed_count + error_count) % 500 == 0 and (processed_count + error_count) > 0:
             print(f"  Processed {processed_count + error_count}/{len(image_paths)} images...")
    end_time = time.time()
    print(f"Dataset embedding generation complete. Processed: {processed_count}, Failed/Skipped: {error_count}, Time: {end_time - start_time:.2f}s")
    if not all_embeddings: 
        return None, None
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    return embeddings_array, all_filepaths

def find_most_similar(input_embedding, dataset_embeddings, dataset_filepaths, input_path=None, top_k=5):
    """Finds the most similar faces based on Euclidean distance."""
    if dataset_embeddings is None or len(dataset_embeddings) == 0: 
        return None, float('inf')
    
    # Debug logging
    print(f"\nSimilarity Matching:")
    print(f"Input embedding shape: {input_embedding.shape}")
    print(f"Dataset embeddings shape: {dataset_embeddings.shape}")
    
    # Check if the input image path exists in the dataset
    input_in_dataset = False
    input_index = -1
    if input_path:
        norm_input_path = os.path.normpath(input_path)
        for idx, path in enumerate(dataset_filepaths):
            if os.path.normpath(path) == norm_input_path:
                input_in_dataset = True
                input_index = idx
                print(f"Input image '{norm_input_path}' is already in the dataset at index {idx}.")
                break
        
        # If input is not in dataset, add it
        if not input_in_dataset and input_path:
            print(f"Adding input image '{norm_input_path}' to the dataset for comparison.")
            # Add the input embedding to the dataset
            dataset_embeddings = np.vstack((dataset_embeddings, input_embedding.reshape(1, -1)))
            dataset_filepaths.append(norm_input_path)
            input_index = len(dataset_filepaths) - 1
    
    # Calculate distances - use a safe approach that doesn't require broadcast
    distances = np.zeros(len(dataset_embeddings))
    for i in range(len(dataset_embeddings)):
        diff = dataset_embeddings[i] - input_embedding
        distances[i] = np.sum(np.square(diff))
    
    # Get sorted indices
    sorted_indices = np.argsort(distances)
    
    # Print top k matches and their distances
    print(f"\nTop {top_k} matches:")
    results = []
    
    # Track if we're skipping the input image itself
    skip_first = False
    
    # Check if the closest match is the input itself
    if input_index >= 0 and len(sorted_indices) > 0 and sorted_indices[0] == input_index:
        print(f"Note: The closest match is the input image itself")
        skip_first = True
    
    # Get the top matches, skipping the input if needed
    start_idx = 1 if skip_first else 0
    top_indices = sorted_indices[start_idx:start_idx+top_k]
    
    for i, idx in enumerate(top_indices):
        if idx < len(dataset_filepaths):  # Safety check
            match_path = dataset_filepaths[idx]
            distance = distances[idx]
            similarity = 1.0 - min(distance, 2.0) / 2.0  # Convert distance to similarity score (0-1)
            print(f"Match {i+1}: {match_path} - Similarity: {similarity:.4f} (Distance: {distance:.4f})")
            results.append((match_path, similarity))
    
    # Return the most similar match and its similarity score
    if results:
        return results[0][0], results[0][1]
    return None, 0.0

def save_cache(embeddings_array, paths_list, embeddings_file, paths_file, cache_dir):
    """Saves the embeddings and paths to cache files."""
    try:
        print(f"Saving updated embeddings to cache: {embeddings_file}")
        os.makedirs(cache_dir, exist_ok=True)
        np.save(embeddings_file, embeddings_array)
        with open(paths_file, 'wb') as f:
            pickle.dump(paths_list, f)
        print(f"Successfully saved {len(paths_list)} embeddings and paths to cache.")
    except Exception as e:
        print(f"Error saving embeddings to cache: {e}")

####################################
# MAIN EXECUTION BLOCK
####################################
def select_model(model_name):
    """Configure settings based on model name selection"""
    global CONFIG
    
    # Copy the base config
    config = dict(CONFIG)
    
    if model_name == "resnet18_gs":
        config["model_type"] = "resnet18"
        config["use_se"] = False
        config["grayscale"] = True
    elif model_name == "resnet18_color":
        config["model_type"] = "resnet18"
        config["use_se"] = False
        config["grayscale"] = False
    elif model_name == "resnet18_se_gs":
        config["model_type"] = "resnet18"
        config["use_se"] = True
        config["grayscale"] = True
    elif model_name == "resnet18_se_color":
        config["model_type"] = "resnet18"
        config["use_se"] = True
        config["grayscale"] = False
    # Handle all EdgeFace variants
    elif model_name.startswith("edgeface_") or model_name in ["edgeface"]:
        config["model_type"] = "edgeface"
        # Extract the variant name (remove 'edgeface_' prefix if present)
        variant = model_name
        if model_name == "edgeface":
            variant = "edgeface_xxs"  # Default to xxs if just "edgeface" is specified
        elif model_name.startswith("edgeface_"):
            variant = model_name.replace("edgeface_", "")
        
        # Special cases for complete model names
        if model_name.startswith("edgeface_") and not variant.startswith("edgeface_"):
            config["edgeface_variant"] = variant
        else:
            config["edgeface_variant"] = model_name
            
        config["grayscale"] = False  # EdgeFace models use color images
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: resnet18_gs, resnet18_color, resnet18_se_gs, resnet18_se_color, edgeface_*")
        
    return config

def process_images(image_to_append, image_to_lookup, config):
    """
    Processes two images: appends one to dataset, finds match for the other.

    Args:
        image_to_append (str): Path to the image to add to the dataset.
        image_to_lookup (str): Path to the image to search for similarities against the updated dataset.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (bool, str, float) indicating (success, match_path, similarity)
               or (False, None, 0.0) on failure.
    """
    # --- Configuration ---
    use_cpu = False
    if config["model_type"] == "edgeface" and "q" in config.get("edgeface_variant", ""):
        use_cpu = True
    elif not torch.cuda.is_available():
        print("CUDA not available, forcing CPU.")
        use_cpu = True
    device = torch.device("cpu" if use_cpu else "cuda")
    print(f"Using device: {device}")

    # --- Load Model & Processor ---
    model_loader = FaceModelLoader(config, device)
    model = model_loader.load_model()
    img_processor = ImageProcessor(config)

    # --- Load or Generate Dataset Embeddings ---
    embeddings_cache_file, paths_cache_file = model_loader.get_cache_filenames()
    dataset_embeddings = None
    dataset_filepaths = None
    dataset_modified = False  # Flag to track if saving is needed

    if os.path.exists(embeddings_cache_file) and os.path.exists(paths_cache_file):
        print(f"Loading embeddings from cache: {embeddings_cache_file}")
        try:
            dataset_embeddings = np.load(embeddings_cache_file)
            with open(paths_cache_file, 'rb') as f:
                dataset_filepaths = pickle.load(f)
            print(f"Loaded {len(dataset_filepaths)} embeddings and paths from cache.")
        except Exception as e:
            print(f"Error loading from cache: {e}. Regenerating embeddings.")
            dataset_embeddings, dataset_filepaths = None, None  # Force regeneration

    if dataset_embeddings is None:
        if not config.get("data_root") or not os.path.isdir(config["data_root"]):
            print(f"Error: data_root '{config.get('data_root')}' not specified or not a directory. Cannot generate embeddings.")
            return False, None, 0.0
        dataset_embeddings, dataset_filepaths = get_all_dataset_embeddings(
            model, img_processor, config["data_root"], device, config
        )
        if dataset_embeddings is not None and len(dataset_filepaths) > 0:
            dataset_modified = True  # Mark as modified to ensure saving

    # --- Process Image 1 (Append) ---
    print(f"\n--- Processing Image to Append: {image_to_append} ---")
    # Get normalized path
    norm_append_path = os.path.normpath(image_to_append) if image_to_append else None
    
    # Check if already in dataset
    if norm_append_path and norm_append_path in [os.path.normpath(p) for p in dataset_filepaths]:
        print(f"Image '{image_to_append}' is already in the dataset. Skipping append.")
    elif norm_append_path:
        print(f"Appending '{image_to_append}' to dataset...")
        append_processed = img_processor.process_img(image_to_append)
        if append_processed is not None:
            append_embedding = get_embedding(model, append_processed, device, config)
            if append_embedding is not None:
                # Add embedding and path
                dataset_embeddings = np.vstack((dataset_embeddings, append_embedding.reshape(1, -1)))
                dataset_filepaths.append(norm_append_path)
                dataset_modified = True  # Mark for saving
                print(f"Successfully processed and appended. Dataset size: {len(dataset_filepaths)}")
            else:
                print(f"Failed to generate embedding for '{image_to_append}'. Not added.")
        else:
            print(f"Failed to process '{image_to_append}'. Not added.")

    # --- Process Image 2 (Lookup) ---
    print(f"\n--- Processing Image to Lookup: {image_to_lookup} ---")
    lookup_img_for_processing = image_to_lookup  # Start with the path
    norm_lookup_path = os.path.normpath(image_to_lookup) if image_to_lookup else None

    # Process the lookup image
    lookup_processed = img_processor.process_img(lookup_img_for_processing)
    if lookup_processed is None:
        print(f"Failed to process lookup image: {lookup_img_for_processing}")
        return False, None, 0.0
    
    lookup_embedding = get_embedding(model, lookup_processed, device, config)
    if lookup_embedding is None:
        print(f"Failed to generate embedding for lookup image.")
        return False, None, 0.0
    
    # Ensure the lookup image is also in the dataset for better display
    if norm_lookup_path and norm_lookup_path not in [os.path.normpath(p) for p in dataset_filepaths]:
        print(f"Adding lookup image to dataset for comprehensive results...")
        dataset_embeddings = np.vstack((dataset_embeddings, lookup_embedding.reshape(1, -1)))
        dataset_filepaths.append(norm_lookup_path)
        dataset_modified = True
    
    # Find the most similar image
    most_similar_path, similarity = find_most_similar(
        lookup_embedding, dataset_embeddings, dataset_filepaths, 
        input_path=norm_lookup_path, top_k=1
    )
    
    # Save updated embeddings if changed
    if dataset_modified:
        save_cache(
            dataset_embeddings, dataset_filepaths, 
            embeddings_cache_file, paths_cache_file, 
            os.path.dirname(embeddings_cache_file)
        )
    
    return True, most_similar_path, similarity

def list_available_models():
    print("\nAvailable models:")
    print("  1. resnet18_gs      - ResNet18 Face (Grayscale)")
    print("  2. resnet18_color   - ResNet18 Face (Color)")
    print("  3. resnet18_se_gs   - ResNet18 Face with SE (Grayscale)")
    print("  4. resnet18_se_color- ResNet18 Face with SE (Color)")
    print("  5. edgeface_xxs     - EdgeFace XXS")

def main():
    parser = argparse.ArgumentParser(description='Find similar faces using different models')
    parser.add_argument("reference_img", type=str, help="Path to reference image")
    parser.add_argument("query_img", type=str, nargs='?', help="Path to query image (optional)")
    parser.add_argument("--model", type=str, default="resnet18_gs", help="Model to use")
    parser.add_argument("--list_models", action="store_true", help="List available models")
    parser.add_argument("--data_root", type=str, default="align/lfw-align-128", 
                       help="Root directory of face dataset")
    parser.add_argument("--image_size", type=int, default=128, 
                       help="Image size for processing (square)")
    parser.add_argument("--threshold", type=float, default=0.6, 
                       help="Similarity threshold (0.0-1.0)")
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.isfile(args.reference_img):
        parser.error(f"Reference image not found: {args.reference_img}")
    if args.query_img and not os.path.isfile(args.query_img):
        parser.error(f"Query image not found: {args.query_img}")

    # Process images if we have both reference and query
    if args.reference_img and args.query_img:
        # Configure model and parameters
        config = select_model(args.model)
        config["data_root"] = args.data_root
        config["image_size"] = args.image_size
        config["threshold"] = args.threshold

        # Process images and find matches
        success, match_path, similarity = process_images(
            args.reference_img, 
            args.query_img, 
            config
        )

        # Final result summary
        if success:
            print("\nSimilarity Search Complete!")
            print(f"Model:          {args.model}")
            print(f"Query Image:    {args.query_img}")
            print(f"Matched Image:  {match_path}")
            print(f"Similarity:     {similarity:.4f}")
        else:
            print("\nSimilarity search did not find a conclusive match.")
            # Exit with error code if no match found
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        list_available_models()
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1) 