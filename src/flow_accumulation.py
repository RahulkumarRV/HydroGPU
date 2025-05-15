import cupy as cp

def scatter_add(arr, indices, values):
    """ 
    Performs scatter-add operation for CuPy.
    Equivalent to NumPy's `np.add.at()` but using atomic operations.
    """
    if len(indices) > 0:  # Ensure there are valid updates
        arr[indices[:, 0], indices[:, 1]] += values


def transfer_values_gpu_cupy(v, d):
    """
    Transfers values from matrix `v` to neighboring cells based on direction matrix `d` using CuPy.
    
    v: CuPy array of shape (H, W) containing the values.
    d: CuPy array of shape (H, W) containing direction indices (1-8).
    
    Returns:
        CuPy array of shape (H, W) after value transfer.
    """
    H, W = v.shape

    # Define shift directions corresponding to 1-8 direction mapping
    shift_offsets = {
        1: (-1, 1),   # Top-right
        2: (-1, 0),   # Top-middle
        3: (-1, -1),  # Top-left
        4: (0, -1),   # Left
        5: (1, -1),   # Bottom-left
        6: (1, 0),    # Bottom-middle
        7: (1, 1),    # Bottom-right
        8: (0, 1)     # Right
    }

    result = cp.zeros_like(v)  # Initialize result array

    # Get all valid indices where v > 0
    indices = cp.argwhere(v > 0)  

    for direction, (di, dj) in shift_offsets.items():
        mask = d[indices[:, 0], indices[:, 1]] == direction  # Get mask for the direction
        selected_indices = indices[mask]  # Select only relevant indices

        if len(selected_indices) > 0:  # Avoid empty updates
            # Compute new positions
            new_pos = selected_indices + cp.array([di, dj])

            # Ensure new positions are within bounds
            valid_mask = (new_pos[:, 0] >= 0) & (new_pos[:, 0] < H) & (new_pos[:, 1] >= 0) & (new_pos[:, 1] < W)
            new_pos = new_pos[valid_mask]
            selected_indices = selected_indices[valid_mask]

            # Perform scatter-add operation
            scatter_add(result, new_pos, v[selected_indices[:, 0], selected_indices[:, 1]])

    return result

def iterative_transfer_cupy(v, d):
    """
    Iteratively applies transfer_values_gpu_cupy until v becomes all zeros,
    summing up all intermediate matrices to get the final cumulative result.
    
    v: Initial CuPy matrix (H, W)
    d: Direction CuPy matrix (H, W)
    
    Returns:
        Summed CuPy matrix after all iterations.
    """
    sum_v = cp.zeros_like(v)  # Initialize sum matrix
    i = 0
    while cp.sum(v) > 0:  # Continue until all values are transferred
        v = transfer_values_gpu_cupy(v, d)
        print(cp.sum(v))
        sum_v += v  # Accumulate results
        i += 1

    return sum_v

def find_max_coordinate(matrix):
    """Returns the coordinates of the maximum value in a CuPy matrix."""
    max_idx = cp.argmax(matrix)  # Get the flattened index of the max value
    max_coord = cp.unravel_index(max_idx, matrix.shape)  # Convert to 2D coordinates
    return max_coord

def save_matrix_with_coordinates(matrix, filename):
    """
    Saves the matrix values along with their x, y coordinates to a file.
    
    Parameters:
    matrix (cupy.ndarray): The matrix to save.
    filename (str): The file name to save the data.
    """
    y, x = cp.where(matrix != 0)  # Get nonzero coordinates
    values = matrix[y, x]
    
    data = cp.stack((x, y, values), axis=1)  # Stack x, y, and values
    
    cp.savetxt(filename, data.get(), fmt='%d', delimiter=',', header='x,y,value', comments='')

# Example Usage with TIFF Data
import numpy as np
from flow_direction import load_tif_image
from make_tif import GeoTIFFHandler

handler = GeoTIFFHandler('../tifs/dems/lower_ganga_fd_.tif')

# Load data as NumPy and convert to CuPy
d = cp.asarray(load_tif_image('../tifs/dems/lower_ganga_fd_.tif'))
v = cp.where(d != 0, cp.ones_like(d), cp.zeros_like(d))

# Compute iterative transfer
transferred_v= iterative_transfer_cupy(v, d)

# save_matrix_with_coordinates(transferred_v, 'dist.txt')


# Convert back to NumPy for saving
handler.save_tiff(cp.asnumpy(transferred_v).astype(np.float32), '../tifs/local/debug_lgb_facc.tif')
