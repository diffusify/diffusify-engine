import torch
import math
import itertools
import torch.nn.functional as F
from threading import Event
from typing import Optional

@torch.inference_mode()
def tiled_scale(
    image: torch.Tensor,
    function,
    tile_x: int = 512,
    tile_y: int = 512,
    overlap: int = 32,
    upscale_amount: int = 4,
    output_device: str = "cpu",
    pbar = None,
    interrupt_flag: Optional[Event] = None
) -> torch.Tensor:
    """
    Process an image in tiles with immediate CPU offloading.
    
    Args:
        image: Input tensor of shape [B, C, H, W]
        function: Model inference function
        tile_x: Tile width
        tile_y: Tile height
        overlap: Overlap between tiles
        upscale_amount: Upscaling factor
        output_device: Device to store output tensor
        pbar: Optional progress bar
    """
    batch_size, channels, height, width = image.shape
    output_shape = [
        batch_size,
        channels,
        round(height * upscale_amount),
        round(width * upscale_amount)
    ]
    
    # Handle case where image fits in single tile
    if height <= tile_y and width <= tile_x:
        return function(image).to(output_device)
    
    # Initialize output tensors
    output = torch.zeros(output_shape, device=output_device)
    output_div = torch.zeros(output_shape, device=output_device)
    
    # Calculate tile positions
    y_positions = range(0, height, tile_y - overlap) if height > tile_y else [0]
    x_positions = range(0, width, tile_x - overlap) if width > tile_x else [0]
    
    # Update progress bar total if it exists
    if pbar is not None:
        pbar.total = batch_size * len(y_positions) * len(x_positions)
        pbar.refresh()
    
    for batch in range(batch_size):
        if interrupt_flag and interrupt_flag.is_set():
            break
        img = image[batch:batch+1]
        for y, x in itertools.product(y_positions, x_positions):
            if interrupt_flag and interrupt_flag.is_set():
                break

            # Calculate tile boundaries
            y_start = max(0, min(height - overlap - 1, y))
            x_start = max(0, min(width - overlap - 1, x))
            y_end = min(y_start + tile_y, height)
            x_end = min(x_start + tile_x, width)
            
            # Extract tile
            tile = img[:, :, y_start:y_end, x_start:x_end]
            
            # Process tile
            processed = function(tile).to(output_device)
            
            # Create feathered mask for blending
            mask = torch.ones_like(processed)
            
            # Feather edges
            feather = round(overlap * upscale_amount)
            if feather > 0:
                for t in range(feather):
                    a = (t + 1) / feather
                    # Vertical edges
                    mask[:,:,t:t+1,:].mul_(a)
                    mask[:,:,-t-1:-t,:].mul_(a)
                    # Horizontal edges
                    mask[:,:,:,t:t+1].mul_(a)
                    mask[:,:,:,-t-1:-t].mul_(a)
            
            # Calculate output position
            y_out_start = round(y_start * upscale_amount)
            x_out_start = round(x_start * upscale_amount)
            
            # Add processed tile to output
            y_out_end = y_out_start + processed.shape[2]
            x_out_end = x_out_start + processed.shape[3]
            
            output[batch:batch+1, :, y_out_start:y_out_end, x_out_start:x_out_end].add_(processed * mask)
            output_div[batch:batch+1, :, y_out_start:y_out_end, x_out_start:x_out_end].add_(mask)

            if pbar is not None:
                pbar.update(1)
    
    # Normalize output
    output = output / (output_div + 1e-8)
    return output

def gaussian_blur(tensor, kernel_size, sigma):
    # Create Gaussian kernel
    device = tensor.device
    channels = tensor.shape[1]
    
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size, device=device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2

    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance)) * \
                     torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / \
                     (2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    # Apply gaussian filter
    return F.conv2d(
        tensor,
        gaussian_kernel,
        padding=kernel_size//2,
        groups=channels
    )