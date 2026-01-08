"""Image preprocessing utilities for OCR"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If image cannot be loaded
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path)
        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def to_grayscale(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """
    Convert image to grayscale.
    
    Args:
        image: PIL Image or OpenCV image array
        
    Returns:
        Grayscale image as numpy array
    """
    if isinstance(image, Image.Image):
        # Convert PIL to OpenCV
        image = pil_to_cv2(image)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_threshold(image: np.ndarray, 
                   method: str = 'otsu',
                   threshold_value: int = 127) -> np.ndarray:
    """
    Apply thresholding to image.
    
    Args:
        image: Grayscale image array
        method: Thresholding method ('otsu', 'binary', 'adaptive')
        threshold_value: Threshold value for binary method
        
    Returns:
        Thresholded image
    """
    if method == 'otsu':
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'binary':
        _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    elif method == 'adaptive':
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return thresh


def denoise_image(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Apply denoising to image.
    
    Args:
        image: Input image array
        method: Denoising method ('bilateral', 'gaussian', 'median')
        
    Returns:
        Denoised image
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        image: Input image array
        alpha: Contrast control (1.0-2.0)
        beta: Brightness control (0-100)
        
    Returns:
        Enhanced image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def resize_image(image: np.ndarray, 
                scale_factor: float = 2.0,
                interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resize image for better OCR accuracy.
    
    Args:
        image: Input image array
        scale_factor: Scale factor for resizing
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def preprocess_image(image_input: Union[str, Path, Image.Image],
                    apply_grayscale: bool = True,
                    apply_denoise: bool = True,
                    apply_threshold_flag: bool = True,
                    enhance_contrast_flag: bool = True,
                    resize_scale: Optional[float] = 2.0,
                    threshold_method: str = 'otsu',
                    denoise_method: str = 'bilateral') -> Image.Image:
    """
    Apply comprehensive preprocessing to improve OCR accuracy.
    
    Args:
        image_input: Image path or PIL Image object
        apply_grayscale: Convert to grayscale
        apply_denoise: Apply denoising
        apply_threshold_flag: Apply thresholding
        enhance_contrast_flag: Enhance contrast
        resize_scale: Scale factor for resizing (None to skip)
        threshold_method: Thresholding method
        denoise_method: Denoising method
        
    Returns:
        Preprocessed PIL Image
    """
    # Load image
    if isinstance(image_input, (str, Path)):
        image = load_image(image_input)
    else:
        image = image_input
    
    # Convert to OpenCV format
    cv_image = pil_to_cv2(image)
    
    # Apply preprocessing steps
    if apply_grayscale:
        cv_image = to_grayscale(cv_image)
    
    if apply_denoise:
        cv_image = denoise_image(cv_image, method=denoise_method)
    
    if enhance_contrast_flag:
        cv_image = enhance_contrast(cv_image)
    
    if resize_scale and resize_scale != 1.0:
        cv_image = resize_image(cv_image, scale_factor=resize_scale)
    
    if apply_threshold_flag:
        cv_image = apply_threshold(cv_image, method=threshold_method)
    
    # Convert back to PIL
    if len(cv_image.shape) == 2:  # Grayscale
        processed_image = Image.fromarray(cv_image, mode='L')
    else:  # Color
        processed_image = cv2_to_pil(cv_image)
    
    return processed_image


def preprocess_for_text_type(image_input: Union[str, Path, Image.Image],
                            text_type: str = 'document') -> Image.Image:
    """
    Apply preprocessing optimized for specific text types.
    
    Args:
        image_input: Image path or PIL Image object
        text_type: Type of text ('document', 'handwritten', 'license_plate', 'receipt')
        
    Returns:
        Preprocessed PIL Image
    """
    if text_type == 'document':
        return preprocess_image(
            image_input,
            apply_grayscale=True,
            apply_denoise=True,
            apply_threshold_flag=True,
            enhance_contrast_flag=True,
            resize_scale=2.0,
            threshold_method='otsu'
        )
    elif text_type == 'handwritten':
        return preprocess_image(
            image_input,
            apply_grayscale=True,
            apply_denoise=True,
            apply_threshold_flag=False,
            enhance_contrast_flag=True,
            resize_scale=2.5,
            denoise_method='bilateral'
        )
    elif text_type == 'license_plate':
        return preprocess_image(
            image_input,
            apply_grayscale=True,
            apply_denoise=True,
            apply_threshold_flag=True,
            enhance_contrast_flag=True,
            resize_scale=3.0,
            threshold_method='adaptive'
        )
    elif text_type == 'receipt':
        return preprocess_image(
            image_input,
            apply_grayscale=True,
            apply_denoise=True,
            apply_threshold_flag=True,
            enhance_contrast_flag=True,
            resize_scale=2.0,
            threshold_method='otsu'
        )
    else:
        raise ValueError(f"Unknown text type: {text_type}")


if __name__ == '__main__':
    # Simple CLI for testing preprocessing
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess image for OCR')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--output', '-o', help='Output path for processed image')
    parser.add_argument('--type', default='document', 
                       choices=['document', 'handwritten', 'license_plate', 'receipt'],
                       help='Text type for optimization')
    
    args = parser.parse_args()
    
    try:
        processed = preprocess_for_text_type(args.image_path, args.type)
        
        if args.output:
            processed.save(args.output)
            print(f"Processed image saved to: {args.output}")
        else:
            output_path = Path(args.image_path).with_suffix('.processed.png')
            processed.save(output_path)
            print(f"Processed image saved to: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)