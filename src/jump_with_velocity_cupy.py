
import cupy as cp
import time

def compute_destination(F, V, cell_size, time_threshold):
    h, w = F.shape

    # Define movement offsets based on flow direction values (1 to 8)
    d_row = cp.array([-1, -1, -1,  0,  1,  1,  1,  0])  # Row offsets
    d_col = cp.array([ 1,  0, -1, -1, -1,  0,  1,  1])  # Column offsets

    # Flatten the matrices
    F_flat = F.ravel()
    V_flat = V.ravel()

    # Compute step size (time required for one movement)
    step_time = cell_size / V_flat

    # Initialize position maps
    idx = cp.arange(h * w)  # Linear indices
    row = idx // w  # Initial row indices
    col = idx % w   # Initial column indices

    # Remaining travel time
    remaining_time = time_threshold - step_time

    while cp.any(remaining_time > 0):
        valid_move = remaining_time > 0  # Identify cells that can move

        if not cp.any(valid_move):  # Stop if no valid moves left
            break

        # **Update the flow direction dynamically from the current cell's destination**
        direction = F_flat[idx[valid_move]] - 1  # Fetch flow direction at the new position

        # Compute next row/col positions
        row[valid_move] += d_row[direction]
        col[valid_move] += d_col[direction]

        # Ensure they stay within bounds
        row = cp.clip(row, 0, h - 1)
        col = cp.clip(col, 0, w - 1)

        # Compute new linear index
        idx = row * w + col

        print('iteration ')
        print(idx.reshape(h, w))

        # Update remaining time
        remaining_time[valid_move] -= step_time[valid_move]
        print(remaining_time)

    return idx.reshape(h, w)


def compute_velocity_from_elevation(E, F, cell_size, n=0.2):
    """
    Compute velocity from elevation data using Manning's equation, considering the given flow direction.

    Parameters:
        E (cp.ndarray): Elevation matrix.
        F (cp.ndarray): Flow direction matrix (values 1 to 8).
        cell_size (float): Grid cell size.
        n (float): Manning's roughness coefficient (default 0.03 for natural terrain).
    
    Returns:
        V (cp.ndarray): Computed velocity matrix.
    """
    h, w = E.shape

    # Define row and column shifts based on flow direction (1 to 8)
    row_shifts = cp.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=cp.int32)  # Top row, middle row, bottom row
    col_shifts = cp.array([1, 0, -1, 1, -1, 1, 0, -1], dtype=cp.int32)

    # Compute target row and column based on F
    target_rows = cp.clip(cp.arange(h)[:, None] + row_shifts[F - 1], 0, h - 1)
    target_cols = cp.clip(cp.arange(w)[None, :] + col_shifts[F - 1], 0, w - 1)

    # Compute elevation difference in the direction of flow
    delta_H = E - E[target_rows, target_cols]

    # Compute slope in the direction of flow
    distances = cp.where((F == 2) | (F == 6) | (F == 4) | (F == 8), cell_size, cell_size * cp.sqrt(2))
    S = cp.maximum(delta_H / distances, 0.05)  # Ensure non-negative slope

    # Approximate hydraulic radius (assuming shallow overland flow)
    R = 0.1  # Assume a small depth of 0.1m

    # Compute velocity using Manning's equation
    # V = (1 / n) * (R ** (2/3)) * (S ** 0.1)
    V = 2.4557754 * cp.sqrt(S)

    # Ensure non-negative velocities
    V = cp.maximum(V, 0)

    return V


def get_destination_indices(flow_direction_matrix, velocity_matrix, cell_size=30, time_threshold=10800):
    rows, cols = flow_direction_matrix.shape
    current_rows, current_cols = cp.indices((rows, cols))
    
    # Mapping of directions (1-8) to row and col shifts
    row_shifts = cp.array([-1, -1, -1,  0,  1,  1,  1,  0])  # Row offsets
    col_shifts = cp.array([ 1,  0, -1, -1, -1,  0,  1,  1])  # Column offsets
    
    current_time = cp.zeros((rows, cols), dtype=cp.float32)
    active_mask = cp.ones((rows, cols), dtype=cp.bool_)
    i = 0
    while cp.any(active_mask):
        # Get the flow direction of the active cells
        directions = flow_direction_matrix[current_rows, current_cols]
        
        # Keep indices within bounds
        next_rows = cp.clip(current_rows + row_shifts[directions - 1], 0, rows - 1)
        next_cols = cp.clip(current_cols + col_shifts[directions - 1], 0, cols - 1)
        
        # Get velocity of next cells without overwriting the velocity matrix
        next_velocities = velocity_matrix[next_rows, next_cols]
        
        # Accumulate time correctly
        current_time += cell_size / next_velocities
        
        # Update active mask to stop processing cells that exceed time threshold
        active_mask &= (current_time <= time_threshold)
        
        # Update only active cells
        current_rows = cp.where(active_mask, next_rows, current_rows)
        current_cols = cp.where(active_mask, next_cols, current_cols)
        i += 1
        print(i)
    
    # Compute row-major indices
    # destination_indices = current_rows * cols + current_cols
    
    return current_rows * cols + current_cols


def get_destination_indices_(flow_direction_matrix, velocity_matrix, cell_size=30, time_threshold=10800):
    rows, cols = flow_direction_matrix.shape
    total_cells = rows * cols

    # Flatten the matrix for row-major indexing
    indices = cp.arange(total_cells)
    flow_direction_flat = flow_direction_matrix.ravel()
    velocity_flat = velocity_matrix.ravel()

    # Row-major shifts
    row_shifts = cp.array([-1, -1, -1,  0,  1,  1,  1,  0]) * cols
    col_shifts = cp.array([ 1,  0, -1, -1, -1,  0,  1,  1])
    shifts = row_shifts + col_shifts

    # Time tracking (flattened)
    current_time = cp.zeros(total_cells, dtype=cp.float32)
    active_mask = cp.ones(total_cells, dtype=cp.bool_)
    current_indices = indices.copy()

    while cp.any(active_mask):
        directions = flow_direction_flat[current_indices]  # Get flow directions
        next_indices = current_indices + shifts[directions - 1]  # Row-major index shifts

        # Ensure indices stay within valid bounds
        valid_mask = (next_indices >= 0) & (next_indices < total_cells)

        # Fetch velocities only for valid indices, avoiding out-of-bounds issues
        next_velocities = cp.where(valid_mask, velocity_flat[next_indices], 1.0)

        # Time update using 1D representation
        current_time[active_mask] += cell_size / next_velocities[active_mask]

        # Stop processing for cells that exceed the time threshold
        active_mask &= (current_time <= time_threshold) & valid_mask

        # Update current indices only for active cells
        current_indices = cp.where(active_mask, next_indices, current_indices)

    return current_indices.reshape(rows, cols)



cell_size = 30.0
threshold = 10800.0

# velocity = compute_velocity_from_elevation(elevation_data, d, 30)

# print(get_destination_indices(d, velocity_matrix=velocity, cell_size=30, time_threshold=300))

import numpy as np
from flow_direction import load_tif_image
from make_tif import GeoTIFFHandler
from plot import create_plot

start_time = time.time()

handler = GeoTIFFHandler('../tifs/masalia/masalia_dem/dem_dir.tif')
d = load_tif_image('../tifs/masalia/masalia_dem/dem_dir_remaped.tif')
elevation_data = cp.asarray(load_tif_image('../tifs/masalia/masalia_dem/dem.tif'))
# velocity = load_tif_image('./tifs/local/velocity_masalia_3h.tif')
# create_plot(velocity, './outputs/local/velocity_masalia_temp')
d = cp.asarray(d, dtype=cp.int32)
# velocity = cp.asarray(velocity)
print(cp.unique(elevation_data))
create_plot(elevation_data.get(), '../tifs/masalia/experiment_pngs/dem.png')
velocity = compute_velocity_from_elevation(elevation_data, d, 30)
create_plot(velocity.get(), '../tifs/masalia/experiment_pngs/velocity.png')
handler.save_tiff(velocity.get(), '../tifs/masalia/experiment_tifs/velocity.tif')

rows, cols = velocity.shape
total_cells = rows * cols
indices = cp.arange(total_cells).reshape((rows, cols))
indices = cp.where( elevation_data > 0, indices, 0)
create_plot(indices.get(), '../tifs/masalia/experiment_pngs/index_masalia.png')

destination_idx = get_destination_indices_(d, velocity_matrix=velocity, cell_size=30, time_threshold=threshold)
end_time = time.time()
print(f'iterative flow transfer time : {end_time - start_time}')  

destination_idx = cp.where( elevation_data > 0, destination_idx, 0)
# print(destination_idx)
create_plot(destination_idx.get(), '../tifs/masalia/experiment_pngs/3h_destination_index_masalia.png')
handler.save_tiff(cp.asnumpy(destination_idx).astype(np.int32), '../tifs/masalia/experiment_tifs/3h_destination_index_masalia.tif')








# Example usage
# v = cp.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]], dtype=cp.float32)
# d = cp.array([
#     [7, 7, 7, 6, 6],
#     [7, 7, 7, 6, 6],
#     [8, 8, 7, 6, 5],
#     [1, 1, 8, 7, 6],
#     [1, 8, 8, 8, 0]
# ], dtype=cp.int64)  # 0 means no transfer

# elevation_data = cp.array([
#     [78, 72, 69, 71, 58],
#     [74, 67, 56, 49, 46],
#     [69, 53, 44, 37, 38],
#     [64, 58, 55, 22, 31],
#     [68, 61, 47, 21, 16]
# ], dtype=cp.float64)
