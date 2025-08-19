from PIL import Image, ImageDraw
import numpy as np
import colorsys

def show_overlay_slice(scan_slice, mask_slice, opacity=0.4):
    """
    Creates an overlay visualization of a tumor segmentation on an MRI slice.
    
    Args:
        scan_slice: 2D array containing the MRI slice data
        mask_slice: 2D array containing the segmentation mask
        opacity: Opacity of the overlay (0.0 to 1.0)
        
    Returns:
        PIL Image with the segmentation overlay
    """
    # Handle empty slices gracefully
    if scan_slice.size == 0 or mask_slice.size == 0:
        return Image.new('RGB', (10, 10), (0, 0, 0))
    
    # Normalize the scan slice to 0-255 range for visualization
    if np.ptp(scan_slice) > 0:  # Prevent division by zero - Using np.ptp() for NumPy 2.0 compatibility
        base = ((scan_slice - scan_slice.min()) / np.ptp(scan_slice) * 255).astype(np.uint8)
    else:
        base = np.zeros_like(scan_slice, dtype=np.uint8)
    
    # Create base image
    try:
        base_img = Image.fromarray(base).convert("RGB")
    except Exception as e:
        # Fallback for irregular shaped data
        print(f"Error creating base image: {e}")
        return Image.new('RGB', (100, 100), (0, 0, 0))
    
    # Create an RGBA mask for the overlay
    # Using a colorful segmentation with different values potentially having different colors
    mask_img = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_img)
    
    # Get unique values in the mask (excluding 0)
    unique_vals = np.unique(mask_slice)
    unique_vals = unique_vals[unique_vals > 0]
    
    # BraTS specific color mapping - use standard BraTS colors
    brats_colors = {
        1: (255, 0, 0),      # Red - Necrotic tumor core
        2: (0, 255, 0),      # Green - Peritumoral edematous/invaded tissue
        3: (0, 0, 255),      # Blue - Enhancing tumor
        4: (255, 255, 0)     # Yellow - Additional class if needed
    }
    
    # If we have segmentation data
    if len(unique_vals) > 0:
        # Use different colors for different segmentation values
        for i, val in enumerate(unique_vals):
            # Use BraTS color scheme if the value is in the standard range (1-4)
            if val in brats_colors:
                r, g, b = brats_colors[val]
            else:
                # Create distinct colors for non-standard classes
                hue = (i * 0.35) % 1.0  # Spread colors around the color wheel
                r, g, b = [int(255 * c) for c in colorsys.hsv_to_rgb(hue, 0.9, 0.9)]
            
            # Get coordinates where the mask equals this value
            y_indices, x_indices = np.where(mask_slice == val)
            
            # Draw pixels for this segmentation class
            for y, x in zip(y_indices, x_indices):
                if 0 <= y < base_img.height and 0 <= x < base_img.width:
                    draw.point((x, y), fill=(r, g, b, int(255 * opacity)))
    
    # Combine the base image with the colored overlay
    result = Image.alpha_composite(base_img.convert("RGBA"), mask_img)
    
    # Add a subtle border to make the visualization more professional
    bordered = ImageOps.expand(result, border=2, fill='white')
    
    return result

# Additional utility functions for multi-view display
try:
    from PIL import ImageOps
except ImportError:
    pass  # Handle missing imports gracefully
