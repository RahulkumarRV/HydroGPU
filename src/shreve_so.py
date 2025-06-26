import cupy as cp
from cupy import RawKernel
from flow_direction import load_tif_image
from make_tif import GeoTIFFHandler
import time


FLOW_DIR_TIF = '../tifs/masalia/qgis/qgis_masalia_fdir.tif'
FLOW_ACC_TIF = '../tifs/masalia/qgis/masalia_flow_acc.tif'
DEM = '../tifs/masalia/masalia_dem/dem.tif'
OUTPUT_STREAM_ORDER_TIF = '../tifs/masalia/experiment_tifs/mws_masalia_debug_10000.tif'


handler = GeoTIFFHandler(FLOW_ACC_TIF)
flowdirection = cp.asarray(load_tif_image(FLOW_DIR_TIF))
flowaccumulation = cp.asarray(load_tif_image(FLOW_ACC_TIF))
dem = cp.asarray(load_tif_image(DEM))
THRESHOLD = 10000


# Directional shifts (D8 encoding: 1–8)
DIRS = cp.array([
    [-1, 1],   # 1: Top-right
    [-1, 0],   # 2: Top-center
    [-1, -1],  # 3: Top-left
    [0, -1],   # 4: Left
    [1, -1],   # 5: Bottom-left
    [1, 0],    # 6: Bottom-center
    [1, 1],    # 7: Bottom-right
    [0, 1],    # 8: Right
], dtype=cp.int32)


def get_indegree(flow_dir):
    """Compute indegree matrix by checking how many cells flow into each cell."""
    indegree = cp.zeros_like(flow_dir, dtype=cp.int32)
    for d in range(1, 9):
        di, dj = DIRS[d - 1]
        # Roll the direction matrix to align upstream flow
        shifted = cp.roll(cp.roll((flow_dir == d).astype(cp.int32), di, axis=0), dj, axis=1)
        indegree += shifted
    return indegree

def get_indegree_from_threshold(flow_dir, flow_accum, threshold):
    """Compute indegree counting only cells with accumulation >= threshold."""
    indegree = cp.zeros_like(flow_dir, dtype=cp.int32)
    for d in range(1, 9):
        di, dj = DIRS[d - 1]
        from_dir = (flow_dir == d) & (flow_accum >= threshold)
        shifted = cp.roll(cp.roll(from_dir.astype(cp.int32), di, axis=0), dj, axis=1)
        indegree += shifted
    return indegree

def stream_order_(flow_dir, active):
    """
    Computes Shreve stream order by pushing values from each cell to its downstream neighbor.
    
    Parameters:
        flow_dir: (H, W) CuPy array of int8 in [1, 8] – D8 flow direction
        active: (H, W) CuPy boolean array – initial active sources

    Returns:
        acc: (H, W) CuPy int32 array – Shreve stream order
    """
    H, W = flow_dir.shape
    acc = cp.zeros((H, W), dtype=cp.int32)
    current = cp.zeros((H, W), dtype=cp.int32)

    # Initialize source cells
    current[active] = 1
    acc[active] = 1

    offsets = DIRS.astype(cp.int32)

    while cp.any(current > 0):
        next_current = cp.zeros((H, W), dtype=cp.int32)

        for d in range(1, 9):
            dy, dx = offsets[d - 1]

            # Identify cells flowing in this direction
            mask = (flow_dir == d)
            source_vals = current * mask

            # Push values downstream: roll source cells into downstream
            shifted = cp.roll(cp.roll(source_vals, dy, axis=0), dx, axis=1)

            # Accumulate and set next wavefront
            acc += shifted
            next_current += shifted

        current = next_current

    return acc

def stream_order(flow_dir, active):
    """
    Optimized Shreve stream order computation using CuPy.

    Parameters:
        flow_dir: (H, W) CuPy array in [1, 8] – D8 flow directions
        active: (H, W) CuPy bool array – source stream cells

    Returns:
        acc: (H, W) CuPy int32 array – Shreve stream order
    """
    H, W = flow_dir.shape
    acc = cp.zeros((H, W), dtype=cp.int32)
    current = cp.zeros((H, W), dtype=cp.int32)
    current[active] = 1
    acc[active] = 1

    offsets = DIRS.astype(cp.int32)
    
    # Precompute direction masks
    masks = [(flow_dir == d) for d in range(1, 9)]

    next_current = cp.zeros((H, W), dtype=cp.int32)

    while cp.count_nonzero(current) > 0:
        next_current.fill(0)  # reuse same memory

        for d in range(8):
            dy, dx = offsets[d]
            source_vals = current * masks[d]  # cells flowing in direction d+1
            shifted = cp.roll(cp.roll(source_vals, dy, axis=0), dx, axis=1)

            acc += shifted
            next_current += shifted

        current, next_current = next_current, current

    return acc

# ------------------------------------------------------

shreve_kernel = RawKernel(r'''
extern "C" __global__
void shreve_push(
    const int* flow_dir, const int* current,
    int* acc, int* next,
    int height, int width)
{
    const int i = blockDim.y * blockIdx.y + threadIdx.y;
    const int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= height || j >= width)
        return;

    const int idx = i * width + j;
    const int val = current[idx];
    const int dir = flow_dir[idx];

    if (val == 0 || dir < 1 || dir > 8)
        return;

    // D8 direction offsets (1-8)
    const int dx[8] = {1, 0, -1, -1, -1, 0, 1, 1};
    const int dy[8] = {-1, -1, -1, 0, 1, 1, 1, 0};

    const int ni = i + dy[dir - 1];
    const int nj = j + dx[dir - 1];

    if (ni < 0 || ni >= height || nj < 0 || nj >= width)
        return;

    const int n_idx = ni * width + nj;

    atomicAdd(&acc[n_idx], val);
    atomicAdd(&next[n_idx], val);
}
''', 'shreve_push')


def stream_order_shreve_kernel(flow_dir, active):
    H, W = flow_dir.shape
    acc = cp.zeros((H, W), dtype=cp.int32)
    current = cp.zeros((H, W), dtype=cp.int32)
    next_buf = cp.zeros((H, W), dtype=cp.int32)

    current[active] = 1
    acc[active] = 1

    threads_per_block = (16, 16)
    blocks_per_grid = ((W + 15) // 16, (H + 15) // 16)

    while cp.any(current):
        next_buf.fill(0)

        shreve_kernel(
            blocks_per_grid, threads_per_block,
            (
                flow_dir.astype(cp.int32).ravel(),
                current.ravel(),
                acc.ravel(),
                next_buf.ravel(),
                H, W
            )
        )

        current, next_buf = next_buf, current  # swap

    return acc


#-------------------------------------------------------

start_time = time.time()
indegree_thresh = get_indegree_from_threshold(flowdirection, flowaccumulation, THRESHOLD)
active = (flowaccumulation >= THRESHOLD) & (indegree_thresh == 0)
stream_order = stream_order_shreve_kernel(flowdirection, active)
stream_order = cp.where(dem != 0, stream_order, 0)
print("indegree calculation time : ", time.time() - start_time)

handler.save_tiff(stream_order.astype(cp.int32).get(), OUTPUT_STREAM_ORDER_TIF)