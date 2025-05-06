#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import argparse
from face_align import align_face, save_aligned_face

def main():
    parser = argparse.ArgumentParser(description="Test face alignment on images")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--output_dir", default="aligned_faces", help="Output directory for aligned faces")
    parser.add_argument("--target_size", type=int, default=128, help="Size of output aligned faces")
    parser.add_argument("--margin", type=float, default=0.3, help="Margin around face as fraction of face size")
    parser.add_argument("--display", action="store_true", help="Display results in window")
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process a single image or all images in a directory
    if os.path.isfile(args.input):
        # Single image
        process_image(args.input, args)
    elif os.path.isdir(args.input):
        # Directory of images
        count = 0
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(args.input, filename)
                result = process_image(input_path, args)
                if result:
                    count += 1
        print(f"Processed {count} images")
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0

def process_image(image_path, args):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    # Get base filename
    base_name = os.path.basename(image_path)
    output_path = os.path.join(args.output_dir, f"aligned_{base_name}")
    
    # Apply face alignment
    aligned_face = align_face(img, args.target_size, args.margin)
    
    if aligned_face is None:
        print(f"Warning: No face detected in {image_path}")
        return False
    
    # Save the aligned face
    cv2.imwrite(output_path, aligned_face)
    print(f"Saved aligned face to {output_path}")
    
    # Display results if requested
    if args.display:
        # Create a side-by-side comparison
        h, w = img.shape[:2]
        max_height = max(h, aligned_face.shape[0])
        
        # Resize original to have same height as combined image
        scale = max_height / h
        resized_original = cv2.resize(img, (int(w * scale), max_height))
        
        # Create combined image
        combined = np.hstack((resized_original, cv2.resize(aligned_face, (max_height, max_height))))
        
        # Show the result
        cv2.imshow(f"Original | Aligned - {image_path}", combined)
        cv2.waitKey(0)
    
    return True

if __name__ == "__main__":
    sys.exit(main()) 