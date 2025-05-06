#!/usr/bin/env python3
"""
Integration guide for using face alignment with find_similar_enhanced.py

This script demonstrates how to modify find_similar_enhanced.py to include
face alignment in the processing pipeline. Apply these changes to improve
the accuracy of face recognition with webcam input.
"""

# Add the following import at the top of find_similar_enhanced.py
"""
try:
    from face_align import align_face
    HAS_ALIGNMENT = True
except ImportError:
    print("Warning: Could not import face_align.py.")
    print("Make sure the file is in the same directory or Python path.")
    HAS_ALIGNMENT = False
"""

# Modify the ImageProcessor.process_img method to include face alignment
"""
def process_img(self, img_input):
    # ... existing code ...
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

        # --- Apply Face Alignment if Available ---
        if HAS_ALIGNMENT:
            try:
                # Only align when we have input directly from camera/uploaded file
                if isinstance(img_input, np.ndarray):
                    aligned_img = align_face(img, target_size=self.image_size)
                    if aligned_img is not None:
                        print("Applied face alignment")
                        img = aligned_img
                    else:
                        print("Face alignment failed, using original image")
            except Exception as e:
                print(f"Error in face alignment: {e}")
        
        # --- If not aligned or if alignment failed, resize image ---
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
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
"""

# Also modify the process_images function to handle live camera input
"""
def process_images(image_to_append, image_to_lookup, config):
    # ... existing code ...
    
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
            # ... existing code ...
    
    # --- Process Image 2 (Lookup) ---
    print(f"\n--- Processing Image to Lookup: {image_to_lookup} ---")
    lookup_img_for_processing = image_to_lookup
    
    # Check if it's a numpy array (likely from webcam) and needs alignment
    if isinstance(image_to_lookup, np.ndarray) and HAS_ALIGNMENT:
        try:
            aligned_img = align_face(image_to_lookup, target_size=config["image_size"])
            if aligned_img is not None:
                print("Applied face alignment to webcam/uploaded image")
                # Need to save to temporary file for processing
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, aligned_img)
                    lookup_img_for_processing = tmp_path
        except Exception as e:
            print(f"Error aligning face: {e}")
    
    # ... rest of process_images function ...
"""

# Integration instructions
print("""
Integration Instructions:
------------------------
1. Make sure to copy face_align.py to the same directory as find_similar_enhanced.py
2. Add the new import at the top of the file
3. Modify the ImageProcessor.process_img method as shown above
4. Update the process_images function if handling numpy arrays from webcam input

These changes will integrate face alignment into the processing pipeline,
improving the accuracy of facial recognition with webcam input.
""") 