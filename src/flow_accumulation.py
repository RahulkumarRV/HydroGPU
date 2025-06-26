import cupy as cp
from make_tif import GeoTIFFHandler
from flow_direction import load_tif_image
import numpy as np
import time

# Define direction offsets mapping 1-8 to (row_offset, col_offset)
# Note: In rasterio/numpy/cupy, rows are typically the first dimension (y),
# and columns the second (x).
# The mapping from the original TF code seems to follow a grid like this:
# 3  2  1
# 4     8
# 5  6  7
# (di, dj)
# (row, col) offset where positive row is down, positive col is right.
# So:
# 1: Top-right -> row -1, col +1
# 2: Top-middle -> row -1, col 0
# 3: Top-left -> row -1, col -1
# 4: Left -> row 0, col -1
# 5: Bottom-left -> row +1, col -1
# 6: Bottom-middle -> row +1, col 0
# 7: Bottom-right -> row +1, col +1
# 8: Right -> row 0, col +1
# This matches the original shift_offsets dictionary.


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

# def transfer_values_cupy_bincount(v, d):
#     """
#     Transfers values from matrix `v` to neighboring cells based on direction matrix `d` using CuPy and bincount.

#     v: CuPy array of shape (H, W) containing the values to transfer in this step.
#     d: CuPy array of shape (H, W) containing direction indices (1-8).

#     Returns:
#         CuPy array of shape (H, W) representing the values transferred IN THIS STEP.
#     """
#     H, W = v.shape
#     # Use float32 for accumulating values

#     # *** GUARANTEE result is assigned at the start ***
#     result = cp.zeros((H, W), dtype=cp.float32)

#     # Find cells that have a value > 0 to transfer
#     has_value_mask = (v > 0)

#     src_rows_all, src_cols_all = cp.nonzero(has_value_mask)
#     # Only proceed if there are any values > 0 to potentially transfer
#     if src_rows_all.size == 0:
#         return result # No values to transfer, return the initial zero result

#     values_to_transfer_all = v[src_rows_all, src_cols_all]
#     directions_all = d[src_rows_all, src_cols_all]

#     # Filter for valid directions (1-8) in the sources
#     valid_direction_mask = (directions_all >= 1) & (directions_all <= 8)
#     src_rows_valid = src_rows_all[valid_direction_mask]
#     src_cols_valid = src_cols_all[valid_direction_mask]
#     values_valid = values_to_transfer_all[valid_direction_mask]
#     directions_valid = directions_all[valid_direction_mask]

#     # Check if there are any valid transfers towards valid directions
#     if values_valid.size > 0:
#         all_flat_dest_indices = []
#         all_transfer_values = []

#         for direction, (di, dj) in shift_offsets.items():
#              direction_mask_this_dir = (directions_valid == direction)
#              if not cp.any(direction_mask_this_dir):
#                  continue

#              src_rows_this_dir = src_rows_valid[direction_mask_this_dir]
#              src_cols_this_dir = src_cols_valid[direction_mask_this_dir]
#              values_this_dir = values_valid[direction_mask_this_dir]

#              dest_rows_this_dir = src_rows_this_dir + di
#              dest_cols_this_dir = src_cols_this_dir + dj

#              # Check boundary conditions
#              valid_mask_this_dir = (dest_rows_this_dir >= 0) & (dest_rows_this_dir < H) & \
#                                     (dest_cols_this_dir >= 0) & (dest_cols_this_dir < W)

#              valid_dest_rows = dest_rows_this_dir[valid_mask_this_dir]
#              valid_dest_cols = dest_cols_this_dir[valid_mask_this_dir]
#              valid_values_this_dir = values_this_dir[valid_mask_this_dir] # Renamed to avoid conflict if needed

#              if valid_values_this_dir.size > 0:
#                 # Flatten valid destination indices
#                 flat_valid_dest_indices = valid_dest_rows * W + valid_dest_cols

#                 # Collect flattened indices and values from all directions
#                 all_flat_dest_indices.append(flat_valid_dest_indices)
#                 all_transfer_values.append(valid_values_this_dir) # Append renamed variable


#         # *** Perform bincount only if any transfers were collected ***
#         if all_flat_dest_indices:
#              all_flat_dest_indices = cp.concatenate(all_flat_dest_indices)
#              all_transfer_values = cp.concatenate(all_transfer_values)

#              bincount_result_flat = cp.bincount(
#                  all_flat_dest_indices,
#                  weights=all_transfer_values,
#                  minlength=H * W
#              )

#              # *** Reassign result with the calculated transfers ***
#              result = bincount_result_flat.reshape(H, W)
#         # Note: If all_flat_dest_indices is empty, result remains the zeros matrix initialized at the start.

#     # If values_valid.size was initially 0, result also remains the zeros matrix.
#     # The initial check 'if src_rows_all.size == 0:' also handles the case
#     # where there are no values > 0 to begin with.

#     return result # result is now guaranteed to be assigned


def transfer_values_optimized(v, d):
    """
    Efficient in-place value transfer using CuPy.
    
    v: CuPy array of shape (H, W)
    d: CuPy array of shape (H, W)
    
    Returns:
        CuPy array (H, W) of transferred values.
    """
    H, W = v.shape
    result = cp.zeros_like(v, dtype=cp.float32)

    for dir_val, (dy, dx) in shift_offsets.items():
        # Find cells with value > 0 and direction == dir_val
        mask = (v > 0) & (d == dir_val)

        if not cp.any(mask):
            continue

        # Get source indices
        src_rows, src_cols = cp.nonzero(mask)
        values = v[src_rows, src_cols]

        # Compute destination indices
        dst_rows = src_rows + dy
        dst_cols = src_cols + dx

        # Filter out of bounds
        valid = (dst_rows >= 0) & (dst_rows < H) & (dst_cols >= 0) & (dst_cols < W)
        dst_rows = dst_rows[valid]
        dst_cols = dst_cols[valid]
        values = values[valid]

        # Flattened destination index for bincount
        dst_flat = dst_rows * W + dst_cols
        acc = cp.bincount(dst_flat, weights=values, minlength=H * W)

        # Reshape and accumulate in result
        result += acc.reshape((H, W))

    return result


def iterative_transfer_cupy(v, d):
    """
    Iteratively applies transfer_values_cupy until v becomes all zeros,
    summing up all intermediate matrices to get the final cumulative result.

    v: Initial CuPy array (H, W) of values to transfer.
    d: Direction CuPy array (H, W).

    Returns:
        Summed CuPy array after all iterations.
    """
    # Ensure v is float for summation
    v = v.astype(cp.float32)
    sum_v = cp.zeros_like(v, dtype=cp.float32)  # Initialize sum matrix
    # Loop while there are still values > 0 in v to transfer
    while cp.any(v > 0):
        # Compute the values transferred in this step
        new_v = transfer_values_optimized(v, d)

        # Accumulate the results of this step into the total sum
        sum_v = sum_v + new_v

        # The values for the next iteration are the values that were just transferred
        v = new_v

    print("Iterative transfer finished.")
    return sum_v



# --- Main Execution ---

## Inputes 
FLOW_DIR_TIF = '../tifs/lower_ganga_basin/dem/lower_ganga_fd.tif'
FLOW_ACC_TIF = '../tifs/lower_ganga_basin/dem/full_itr_floacc.tif'
DEM = '../tifs/lower_ganga_basin/dem/lower_ganga_dem.tif'
OUTPUT_STREAM_ORDER_TIF = '../tifs/lower_ganga_basin/flow_acc/lgb_facc.tif'

# Initialize GeoTIFF handler, which will handle reading and writing GeoTIFF files
handler = GeoTIFFHandler(FLOW_DIR_TIF)

# Load data as NumPy and convert to CuPy
dem = cp.asarray(load_tif_image(FLOW_DIR_TIF))
v = cp.where(dem != 0, cp.ones_like(dem), cp.zeros_like(dem))

# Compute iterative transfer
start_time = time.time()
transferred_v= iterative_transfer_cupy(v, dem)
print(f"flow accumulation time : {time.time() - start_time}")

# Ensure transferred_v is zero where dem is zero
# This is to match the original TF code behavior where only valid DEM cells are considered.
transferred_v = cp.where(dem != 0, transferred_v, 0)

# Convert back to NumPy for saving to disk
handler.save_tiff(cp.asnumpy(transferred_v).astype(np.float32),  OUTPUT_STREAM_ORDER_TIF)


print("DEM Preprocessing (Flow Accumulation) Completed with CuPy!")