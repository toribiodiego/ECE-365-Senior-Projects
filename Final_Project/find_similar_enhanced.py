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

####################################
# CONFIGURATION DICTIONARY
####################################

# Example Configuration (Adjust as needed)
config = {
    "pretrained_path": "resnet18_110.pth",  # Not used for EdgeFace.
    "model_type": "edgeface",               # Options: "resnet18" or "edgeface"
    "edgeface_variant": "edgeface_xxs",     # Quantized variant; force CPU if "q" in name
    "use_se": False,                        # Whether to use Squeeze-Excitation blocks (ResNetFace)
    "grayscale": False,                     # For EdgeFace, use color images (True for ResNetFace default)
    "image_size": 128,                      # Input image size (width, height)
    "data_root": "align/lfw-align-128",      # Root directory of dataset images for INITIAL search/generation
    "embedding_size": 512,                  # Expected embedding dimension from the model
    "cache_dir": "embeddings_cache"         # Directory to store cached embeddings
}

####################################
# 1) MODEL DEFINITION (ResNetFace) - Keep as is
####################################
# (ResNetFace, IRBlock, SEBlock, conv3x3 definitions remain the same)
# ... [Keep the existing ResNetFace, IRBlock, SEBlock, conv3x3 definitions here] ...
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
    def __init__(self, block, layers, use_se=True, grayscale=True):
        super(ResNetFace, self).__init__()
        self.inplanes = 64
        in_ch = 1 if grayscale else 3
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], use_se=use_se) # Pass use_se here
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_se=use_se) # Pass use_se here
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_se=use_se) # Pass use_se here
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_se=use_se) # Pass use_se here
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        # Adjust the input size to fc5 based on the actual output size after layers
        # If image_size is 128, after maxpool and 3 stride-2 layers, it becomes 128 / 2 / 2 / 2 / 2 = 8
        fc_input_size = 512 * (config["image_size"] // 16) * (config["image_size"] // 16)
        self.fc5 = nn.Linear(fc_input_size, config["embedding_size"])
        self.bn5 = nn.BatchNorm1d(config["embedding_size"])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True): # Add use_se default
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se)) # Pass use_se
        self.inplanes = planes * block.expansion # Correctly update inplanes for expansion blocks if used
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se)) # Pass use_se
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
        # Return the embedding directly for simplicity, or dict if needed later
        return x # Directly return the tensor

def resnet_face18(use_se=True, grayscale=True):
    # Use the config directly for use_se
    return ResNetFace(IRBlock, [2, 2, 2, 2], use_se=config["use_se"], grayscale=grayscale)

####################################
# 2) MODEL LOADER CLASS - Keep as is
####################################
class FaceModelLoader:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def load_model(self):
        if self.config["model_type"] == "resnet18":
            is_grayscale = self.config.get("grayscale", True)
            model = resnet_face18(use_se=self.config["use_se"], grayscale=is_grayscale).to(self.device)
            try:
                state_dict = torch.load(self.config["pretrained_path"], map_location=self.device)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                print(f"Loaded ResNetFace model from {self.config['pretrained_path']}")
            except FileNotFoundError:
                 print(f"Warning: Pretrained path {self.config['pretrained_path']} not found for ResNetFace. Model is initialized randomly.")
            except Exception as e:
                 print(f"Error loading ResNetFace state dict: {e}. Model is initialized randomly.")
            model.eval()
            return model

        elif self.config["model_type"] == "edgeface":
            try:
                model = torch.hub.load(
                    repo_or_dir='otroshi/edgeface', model=self.config["edgeface_variant"],
                    source='github', pretrained=True
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

####################################
# 3) IMAGE PROCESSOR CLASS - Keep as is
####################################
class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.is_grayscale = self.config.get("grayscale", False)
        if self.config["model_type"] == "edgeface": self.is_grayscale = False
        elif self.config["model_type"] == "resnet18": self.is_grayscale = self.config.get("grayscale", True)
        print(f"ImageProcessor initialized. Grayscale: {self.is_grayscale}")

    def process_img(self, img_path):
        try:
            if self.is_grayscale: img = cv2.imread(img_path, 0)
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None: return None
            size = self.config["image_size"]
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            if self.is_grayscale and len(img.shape) == 2: img = np.expand_dims(img, axis=-1)
            if not self.is_grayscale and img.shape[-1] == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.transpose((2, 0, 1)).astype(np.float32)
            img = (img - 127.5) / 127.5
            return torch.from_numpy(img).float()
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

####################################
# 4) HELPER FUNCTIONS FOR INFERENCE & CACHING - Keep as is
####################################

def get_embedding(model, processed_img_tensor, device, config):
    """Gets embedding for a single processed image tensor."""
    if processed_img_tensor is None: return None
    processing_device = torch.device("cpu") if ("q" in config.get("edgeface_variant", "") and config["model_type"] == "edgeface") else device
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
            else: error_count += 1
        else: error_count += 1
        if (processed_count + error_count) % 500 == 0 and (processed_count + error_count) > 0:
             print(f"  Processed {processed_count + error_count}/{len(image_paths)} images...")
    end_time = time.time()
    print(f"Dataset embedding generation complete. Processed: {processed_count}, Failed/Skipped: {error_count}, Time: {end_time - start_time:.2f}s")
    if not all_embeddings: return None, None
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    embeddings_array = normalize(embeddings_array, axis=1, norm='l2')
    return embeddings_array, all_filepaths

def find_most_similar(input_embedding, dataset_embeddings, dataset_filepaths, input_path=None, top_k=1):
    """Find the most similar face in the dataset based on Euclidean distance."""
    # Compute distances between input embedding and all dataset embeddings
    distances = np.linalg.norm(dataset_embeddings - input_embedding, axis=1)
    
    # Sort distances
    sorted_indices = np.argsort(distances)
    
    # Print top 5 matches and their distances
    print("\nTop 5 matches:")
    for i in range(min(5, len(distances))):
        idx = sorted_indices[i]
        print(f"Match {i+1}: {dataset_filepaths[idx]} - Distance: {distances[idx]:.4f}")
    
    min_dist_idx = sorted_indices[0]
    min_dist = distances[min_dist_idx]
    
    # Check if the closest match is the input (lookup) image itself
    if input_path:
        norm_input_path = os.path.normpath(input_path)
        norm_match_path = os.path.normpath(dataset_filepaths[min_dist_idx])
        print(f"\nPath comparison:")
        print(f"Input path (normalized): {norm_input_path}")
        print(f"Match path (normalized): {norm_match_path}")
        print(f"Paths equal: {norm_input_path == norm_match_path}")
        
        if norm_input_path == norm_match_path:
            print("Closest match is the lookup image itself.")
            if len(distances) > 1:
                second_min_dist_idx = sorted_indices[1]
                second_min_dist = distances[second_min_dist_idx]
                print(f"Found second closest match.")
                return dataset_filepaths[second_min_dist_idx], second_min_dist
            else:
                print("Only one image in dataset (the input itself), cannot find another match.")
                return dataset_filepaths[min_dist_idx], min_dist
    
    return dataset_filepaths[min_dist_idx], min_dist

def get_cache_filenames(config):
    """Generates filenames for cached embeddings based on config."""
    model_id = config["model_type"]
    if config["model_type"] == "edgeface": model_id += "_" + config["edgeface_variant"]
    model_id += f"_img{config['image_size']}"
    base_filename = f"cache_{model_id}"
    embeddings_file = os.path.join(config["cache_dir"], base_filename + ".npy")
    paths_file = os.path.join(config["cache_dir"], base_filename + "_paths.pkl")
    return embeddings_file, paths_file

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
# 5) MAIN EXECUTION BLOCK
####################################
def test_all_models(test_image_path, data_root, device):
    """Test all available models with the given test image."""
    # Define all model configurations to test
    model_configs = [
        # EdgeFace variants - only testing xxs as it's the only one available
        {
            "model_type": "edgeface",
            "edgeface_variant": "edgeface_xxs",
            "grayscale": False,
            "image_size": 128,
            "pretrained_path": None,  # EdgeFace loads from hub
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        },
        # ResNetFace variants
        {
            "model_type": "resnet18",
            "use_se": True,
            "grayscale": True,
            "image_size": 128,
            "pretrained_path": "resnet18_110.pth",
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        },
        {
            "model_type": "resnet18",
            "use_se": False,
            "grayscale": True,
            "image_size": 128,
            "pretrained_path": "resnet18_110.pth",
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        },
        {
            "model_type": "resnet18",
            "use_se": True,
            "grayscale": False,
            "image_size": 128,
            "pretrained_path": "resnet18_110.pth",
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        },
        {
            "model_type": "resnet18",
            "use_se": False,
            "grayscale": False,
            "image_size": 128,
            "pretrained_path": "resnet18_110.pth",
            "embedding_size": 512,
            "cache_dir": "embeddings_cache"
        }
    ]

    results = {}
    for model_config in model_configs:
        try:
            print(f"\nTesting model configuration: {model_config}")
            
            # Create model loader with current config
            model_loader = FaceModelLoader(model_config, device)
            model = model_loader.load_model()
            
            # Create image processor
            img_processor = ImageProcessor(model_config)
            
            # Process test image
            test_img = img_processor.process_img(test_image_path)
            if test_img is None:
                print(f"Failed to process test image with {model_config}")
                continue
            
            # Get test image embedding
            test_embedding = get_embedding(model, test_img, device, model_config)
            if test_embedding is None:
                print(f"Failed to get embedding for test image with {model_config}")
                continue
            
            # Get dataset embeddings
            dataset_embeddings, dataset_filepaths = get_all_dataset_embeddings(
                model, img_processor, data_root, device, model_config
            )
            
            # Check if test image is in the dataset, add it if not
            norm_test_path = os.path.normpath(test_image_path)
            test_in_dataset = False
            for path in dataset_filepaths:
                if os.path.normpath(path) == norm_test_path:
                    test_in_dataset = True
                    print(f"Test image '{norm_test_path}' is already in the dataset.")
                    break
            
            # Add test image to dataset if not found
            if not test_in_dataset:
                print(f"Adding test image '{norm_test_path}' to the dataset for better comparison.")
                dataset_embeddings = np.vstack((dataset_embeddings, test_embedding.reshape(1, -1)))
                dataset_filepaths.append(norm_test_path)
                
                # Save updated embeddings
                embeddings_file, paths_file = get_cache_filenames(model_config)
                save_cache(dataset_embeddings, dataset_filepaths, embeddings_file, paths_file, model_config["cache_dir"])
            
            # Find most similar
            similar_path, similarity = find_most_similar(
                test_embedding, dataset_embeddings, dataset_filepaths, test_image_path
            )
            
            # Store results
            model_name = f"{model_config['model_type']}_{model_config.get('edgeface_variant', '')}_{model_config.get('use_se', '')}_{model_config.get('grayscale', '')}"
            results[model_name] = {
                'similar_path': similar_path,
                'similarity': similarity
            }
            
            print(f"Results for {model_name}:")
            print(f"Most similar image: {similar_path}")
            print(f"Similarity score: {similarity}")
            
        except Exception as e:
            print(f"Error testing model configuration {model_config}: {e}")
            continue
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find similar images using different face recognition models')
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
    for model_name, result in results.items():
        print(f"\nModel: {model_name}")
        print(f"Most similar image: {result['similar_path']}")
        print(f"Similarity score: {result['similarity']}")
        print("-" * 80)