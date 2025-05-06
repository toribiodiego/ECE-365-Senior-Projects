import cv2
import numpy as np
import os

# Initialize face detector
face_detector = None

def get_face_detector():
    """Get or initialize the face detector."""
    global face_detector
    if face_detector is None:
        # Try to load the face detector - using Haar Cascade for simplicity
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
        ]
        
        for cascade_path in cascade_paths:
            if os.path.exists(cascade_path):
                try:
                    face_detector = cv2.CascadeClassifier(cascade_path)
                    # Verify that the cascade loaded correctly
                    if not face_detector.empty():
                        print(f"Loaded Haar cascade from: {cascade_path}")
                        break
                    else:
                        print(f"Failed to load cascade from: {cascade_path}")
                        face_detector = None
                except Exception as e:
                    print(f"Error loading cascade from {cascade_path}: {e}")
                    face_detector = None
        
        # If all cascade paths failed, create an empty cascade as last resort
        if face_detector is None or face_detector.empty():
            print("WARNING: All face detector initialization attempts failed.")
            print("Creating empty cascade classifier as fallback.")
            face_detector = cv2.CascadeClassifier()
    
    return face_detector

def detect_faces(image):
    """
    Detect faces in the image and return their bounding boxes.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        List of face bounding boxes (x, y, w, h)
    """
    # Input validation
    if image is None or image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        print("Warning: Invalid image passed to detect_faces")
        return []
        
    detector = get_face_detector()
    faces = []
    
    # Check if detector is valid
    if detector is None or detector.empty():
        print("Warning: Face detector not available")
        return []
    
    try:
        # Make grayscale copy of image for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure the gray image is valid
        if gray is None or gray.size == 0:
            print("Warning: Failed to convert image to grayscale")
            return []
            
        # Apply Gaussian blur to reduce noise (helps with detection)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use more conservative parameters for detectMultiScale
        face_rects = detector.detectMultiScale(
            gray, 
            scaleFactor=1.2,      # More conservative scale factor
            minNeighbors=5,       # More reliable detection
            minSize=(30, 30),     # Minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE  # Use image scaling
        )
        
        for (x, y, w, h) in face_rects:
            faces.append((x, y, w, h))
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []
    
    return faces

def align_face(image, target_size=128, margin=0.3):
    """
    Detect, crop, and align a face in the image.
    
    Args:
        image: Input image (BGR format from OpenCV)
        target_size: Size of the output image (square)
        margin: Extra margin around the face as a fraction of face size
        
    Returns:
        Cropped and aligned face image at target_size or None if no face detected
    """
    # Check if image is valid
    if image is None or image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        print("Invalid image passed to align_face")
        return None
        
    # Make a copy to avoid modifying the original
    try:
        img = image.copy()
    except Exception as e:
        print(f"Error making image copy: {e}")
        return None
    
    # Get image dimensions
    img_h, img_w = img.shape[:2]
    
    # Detect faces - with error handling
    try:
        faces = detect_faces(img)
    except Exception as e:
        print(f"Face detection error: {e}")
        faces = []
    
    # If no faces detected, return a center crop as fallback
    if not faces:
        print("No face detected, using center crop")
        # Calculate center crop dimensions (use 2/3 of the image)
        crop_size = min(img_h, img_w) * 2 // 3
        center_x = img_w // 2
        center_y = img_h // 2
        start_x = max(0, center_x - crop_size // 2)
        start_y = max(0, center_y - crop_size // 2)
        end_x = min(img_w, start_x + crop_size)
        end_y = min(img_h, start_y + crop_size)
        
        # Crop and resize
        center_crop = img[start_y:end_y, start_x:end_x]
        if center_crop.size == 0:
            return None
        aligned_img = cv2.resize(center_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return aligned_img
    
    # Get the largest face (assuming it's the main subject)
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Calculate margin in pixels
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    # Calculate crop coordinates with margin
    start_x = max(0, x - margin_x)
    start_y = max(0, y - margin_y)
    end_x = min(img_w, x + w + margin_x)
    end_y = min(img_h, y + h + margin_y)
    
    # Crop the face region
    face_crop = img[start_y:end_y, start_x:end_x]
    
    # Resize to target size
    aligned_img = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return aligned_img

def save_aligned_face(image, output_path, target_size=128):
    """
    Detect, align, and save a face from the input image.
    
    Args:
        image: Input image (BGR format)
        output_path: Path to save the aligned face
        target_size: Size of the output image
        
    Returns:
        Path to saved file or None if processing failed
    """
    aligned_face = align_face(image, target_size)
    
    if aligned_face is None:
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the aligned face
    cv2.imwrite(output_path, aligned_face)
    
    return output_path

# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python face_align.py <input_image_path> [output_image_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "aligned_face.jpg"
    
    # Load image
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not load input image {input_path}")
        sys.exit(1)
    
    # Align and save
    output_path = save_aligned_face(img, output_path)
    
    if output_path:
        print(f"Aligned face saved to {output_path}")
    else:
        print("Failed to process image")
        sys.exit(1) 