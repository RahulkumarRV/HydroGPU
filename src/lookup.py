
import cupy as cp
# import numpy as np
# from flow_direction import load_tif_image
# from make_tif import GeoTIFFHandler
# from plot import create_plot
# import time

# Define the kernel
transfer_kernel = cp.RawKernel(r'''
extern "C" __global__
void transfer_flow(const int* F, const float* V, float* V_out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    int dest = F[i];  // Get destination index
    if (dest >= 0 && dest < size) {
        atomicAdd(&V_out[dest], V[i]);  // Transfer value to destination
    }
}
''', 'transfer_flow')

# Function to execute the kernel
def transfer_flow(F, V):
    F_cp = cp.asarray(F, dtype=cp.int32)
    V_cp = cp.asarray(V, dtype=cp.float32)
    V_out = cp.zeros_like(V_cp)  # Initialize output matrix
    
    size = F_cp.size
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    
    # Launch the kernel
    transfer_kernel((blocks_per_grid,), (threads_per_block,), (F_cp, V_cp, V_out, size))

    return V_out





# ############################### LOWER GANGA EXAMPLE ###############################################

# handler = GeoTIFFHandler('../tifs/dems/lower_ganga_fd_.tif')

# fd = cp.asarray(load_tif_image('../tifs/dems/lower_ganga_fd_.tif'))

# destination = cp.asarray(load_tif_image('../tifs/lower_ganga_basin/itr_3H_jump.tif'))

# V = cp.where(fd > 0, 1, 0)
# create_plot(V.get(), '../outputs/temp/temp.png', max=1, normalize=True, colors=['black', 'white'])
# i = 0
# start_time = time.time()
# while i < 1:
#     V = transfer_flow(destination, V)
#     i += 1
# end_time = time.time()

# print(f'min : {cp.min(V)}, max : {cp.max(V)}')
# print(f'lookup time : {end_time - start_time}')


# V = cp.where(fd > 0, V, 0)
# V = V.get()
# create_plot(V, '../outputs/lower_ganga_basin/lgb_3h_flow_temp.png', max=1, normalize=True, colors=['black', 'white'])

# handler.save_tiff(V, '../tifs/local/lgb_3h_flow_temp.tif')
# np.savetxt("array.txt", V.get(), fmt="%d")




##################################### masalia example ##########################################

# handler = GeoTIFFHandler('../tifs/local/flow_dr.tif') 

# fd = cp.asarray(load_tif_image('../tifs/local/flow_dr.tif'))

# destination = cp.asarray(load_tif_image('../tifs/local/itr_destination_idx_masalia_cupy.tif'))

# V = cp.where(fd > 0, 1, 0)

# i = 0
# start_time = time.time()
# while i < 1:
#     V = transfer_flow(destination, V)
#     i += 1
# end_time = time.time()

# print(f'min : {cp.min(V)}, max : {cp.max(V)}')
# print(f'lookup time : {end_time - start_time}')
# V = cp.where(fd > 0, V, 0)
# V = V.get()
# create_plot(V, '../outputs/local/masalia_3h_flow_temp.png', max=1, normalize=True, colors=['black', 'white'])
# handler.save_tiff(V, './tifs/local/masalia_3h_flow_temp.tif')

















# Example usage
# D = tf.constant([
#     [7, 7, 8],
#     [7, 1, 2],
#     [3, 4, 5]
# ], dtype=tf.int32)  # Destination indices (row-major order)

# V = tf.constant([
#     [10, 20, 30],
#     [40, 50, 60],
#     [70, 80, 90]
# ], dtype=tf.float32)  # Value matrix

# i = 0

# while i < 5:
#     V = transfer_values(D, V)
#     i += 1
#     print(f'iteration : ', i)
#     print(V)


# D = tf.convert_to_tensor(destination, dtype=tf.int32)
# V = tf.where(D > 0, tf.constant(1), tf.constant(0))
# i=0

# # t1 = time.time()
# while i < 2:
#     V = transfer_values(D, V)
#     i += 1
# # t2 = time.time()

# print(f'min  : {tf.reduce_min(V).numpy()}, max : {tf.reduce_max(V).numpy()}')
# # print("time taks in lookup : ", t2 - t1)

# create_plot(V.numpy(), './outputs/local/data_3h.png', max=1, normalize=True)
# np.savetxt("array.txt", V.numpy(), fmt="%d")
# handler.save_tiff(V.numpy(), './tifs/local/data_3h.tif')


