# download datasets from GEE
# preprocess the datasets locally reprojection etc.
# calculate runoff simulation


############################## Import Libraries #################################
import ee
import time
import os
import re
import gc
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import rasterio
import numpy as np
from rasterio.merge import merge
# from gee_processing import GEEDataDownloader
from flow_direction import load_tif_image
# from plot import create_plot
from make_tif import GeoTIFFHandler
import cupy as cp
from runoff_cp import compute_M, sum_tif_images, calculate_runoff, calculate_runoff_cupy
from lookup import transfer_flow


########################### Imports - End ########################################


########################### CONSTANTS ############################################

project_id = 'ee-rahul-kumar'
dataset_id = 'JAXA/GPM_L3/GSMaP/v6/operational'
asset_id = 'projects/ee-rahul-kumar/assets/Lower_ganga'
start_date = '2023-07-01'
end_date = '2023-07-02'
drive_folder = 'Lower_Ganga_3H_Rain_' + start_date + '_' + end_date
dataset_from_gdrive = 'downloaded_GEE_rain'
destination_folder = 'merged_rain_dataset'
log_file = 'logfile.log'


########################### Functions Implementations ###########################

def log_to_file(message, log_file="logfile.log"):
    """
    Simple function to log a message to a file.
    
    Args:
        message (str): The message to log
        log_file (str): Path to the log file (default: application.log)
    """
    
    try:
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"Error writing to log file: {e}")

# Example usage
# log_to_file("Application started")
# log_to_file("Error: Database connection failed", "errors.log")

########################### Download datasets from GEE - Start ###################

# geeDataDownloader = GEEDataDownloader(project_id)
# status = geeDataDownloader.execute_gee_tasks(dataset_id, asset_id, start_date, end_date, drive_folder, destination_folder)

# if status["status"] == "failure":
#     print(f"Error while running GEE tasks : {status["message"]}")
#     exit()

# dataset is downloaded into the destination folder.

########################### Download datasets from GEE - End ###################

full_execution_start_time = time.time()

handler = GeoTIFFHandler('../tifs/masalia/gee_outputs/sr1_masalia_23-07-07_ak.tif')

single_file_loading_time_start = time.time()
sr1 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr1_masalia_23-07-07_ak.tif'), dtype=cp.float32)
single_file_loading_time_end = time.time()
log_to_file(f"single file loading time : {single_file_loading_time_end - single_file_loading_time_start}")
sr2 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr2_masalia_23-07-07_ak.tif'), dtype=cp.float32)
sr3 = cp.asarray(load_tif_image('../tifs/masalia/gee_outputs/sr3_masalia_23-07-07_ak.tif'), dtype=cp.float32)



import os
import time
import cupy as cp
import gc
from flow_direction import load_tif_image
from make_tif import GeoTIFFHandler


def process_images(folder_path, output_folder="runoff_output"):
    """
    Process rainfall images, compute runoff using soil moisture, and save results.
    """
    files = sorted(os.listdir(folder_path))  # Sort files for correct order
    os.makedirs(output_folder, exist_ok=True)
    
    images = []
    P_sum = None
    P5_sum = None
    previous_Runoff = None
    log_flag = True

    global single_3H_runoff_time_start
    global single_3H_runoff_time_end
    global single_M1_calculation_time_start
    global single_M1_calculation_time_end
    global single_runoff_calculation_time_start
    global single_runoff_calculation_time_end

    destination_index = cp.asarray(load_tif_image('../tifs/lower_ganga_basin/itr_3H_jump.tif'))

    for index, file in enumerate(files):  # Process up to 112 images
        if log_flag:
            single_3H_runoff_time_start = time.time()

        # Load image and convert to CuPy array
        img = cp.asarray(load_tif_image(os.path.join(folder_path, file)))
        images.append(img)

        # Initialize summation for first 5 images
        if index == 4:
            P_sum = sum_tif_images(cp.array(images), 2, 5)  # Sum 3 images (2,3,4)
            P5_sum = sum_tif_images(cp.array(images), 0, 5)  # Sum all 5 images

        elif index >= 5:
            # Maintain sliding window: Remove oldest, add newest
            P_sum = P_sum - images[0] + images[-1]
            P5_sum = P5_sum - images[0] + images[-1]

            # Remove first image to keep memory in check
            images.pop(0)

        if index >= 4:

            # Compute runoff
            if previous_Runoff is not None:
                P_sum += previous_Runoff
                P5_sum += previous_Runoff

            # Compute soil moisture
            if log_flag:
                single_M1_calculation_time_start = time.time()

            m1_cp = compute_M(sr1, P5_sum)
            m2_cp = compute_M(sr2, P5_sum)
            m3_cp = compute_M(sr3, P5_sum)

            if log_flag:
                single_M1_calculation_time_end = time.time()
                log_to_file(f"Single M1 time: {single_M1_calculation_time_end - single_M1_calculation_time_start}")

            if log_flag:
                single_runoff_calculation_time_start = time.time()
            R = calculate_runoff(P_sum, P5_sum, m1_cp, m2_cp, m3_cp, sr1, sr2, sr3)
            if log_flag:
                single_runoff_calculation_time_end = time.time()
                log_to_file(f"Single Runoff (R) time: {single_runoff_calculation_time_end - single_runoff_calculation_time_start}")
            # transfer the runoff (R) and store for next iteration
            previous_Runoff = transfer_flow(destination_index, R) 

            # Save runoff result
            handler.save_tiff(cp.asnumpy(R), os.path.join(output_folder, f'runoff_simulation_3H_{index}.tif'))

            if log_flag:
                single_3H_runoff_time_end = time.time()
                log_to_file(f"Single 3 Hour runoff time: {single_3H_runoff_time_end - single_3H_runoff_time_start}")
                log_flag = False  # Reset flag after first loop

            # Free unused memory
            # cp.get_default_memory_pool().free_all_blocks()

    print("Processing complete!")


def process_images_and_track_coordinates(folder_path, coordinate=(100, 100), output_folder="runoff_output"):
    """
    Process rainfall images, compute runoff using soil moisture, and track values at a specific coordinate.
    Returns a time series of runoff values at the specified coordinate.

    Args:
        folder_path: Path to folder containing rainfall TIF files
        coordinate: (x, y) coordinate to track (default: (100, 100))
        output_folder: Folder to save outputs (optional)

    Returns:
        list: Time series of runoff values at the specified coordinate
    """

    # Disable memory pool for device memory (GPU)
    # cp.cuda.set_allocator(None)

    def extract_index(filename):
        match = re.search(r'_(\d+)', filename)
        return int(match.group(1)) if match else -1  # fallback to -1 if not matched
    
    files = sorted(os.listdir(folder_path), key=extract_index)
    os.makedirs(output_folder, exist_ok=True)

    images = []
    P_sum = None
    P5_sum = None
    previous_Runoff = None

    # List to store runoff values at the specified coordinate
    runoff_time_series = []
    point_value = cp.array(0.0)

    # Get destination index once before the loop
    t_start_g = time.time()
    destination_index = cp.asarray(load_tif_image('../tifs/masalia/experiment_tifs/3h_destination_index_masalia.tif'))
    t_end = time.time()
    log_to_file(f"Loading destination index time: {t_end - t_start_g:.4f} seconds")

    for index, file in enumerate(files):
        iteration_start = time.time()
        log_to_file(f"\n--- Processing file {index}: {file} ---")

        # Load image
        t_start = time.time()
        img_path = os.path.join(folder_path, file)
        img_numpy = load_tif_image(img_path)
        t_load_numpy = time.time()
        log_to_file(f"1. Loading image from disk: {t_load_numpy - t_start:.4f} seconds")

        # Convert to CuPy
        img = cp.asarray(img_numpy)
        t_convert_cupy = time.time()
        log_to_file(f"2. Convert to CuPy: {t_convert_cupy - t_load_numpy:.4f} seconds")

        del img_numpy  # Free up memory

        images.append(img)
        t_append = time.time()
        log_to_file(f"3. Append to images list: {t_append - t_convert_cupy:.4f} seconds")

        if index == 4:
            P_sum = cp.sum(cp.stack(images[2:5]), axis=0)
            P5_sum = cp.sum(cp.stack(images[:5]), axis=0)
            log_to_file(f"4. Initial sum creation: {time.time() - t_append:.4f} seconds")

        elif index >= 5:
            t_update_sum_start = time.time()
            P_sum = P_sum - images[0] + images[-1]
            P5_sum = P5_sum - images[0] + images[-1]
            t_update_sum_end = time.time()
            log_to_file(f"4. Update sums: {t_update_sum_end - t_update_sum_start:.4f} seconds")

            t_pop_start = time.time()
            old_img = images.pop(0)
            del old_img
            t_pop_end = time.time()
            log_to_file(f"5. Pop and delete oldest image: {t_pop_end - t_pop_start:.4f} seconds")

        if index >= 4:
            if previous_Runoff is not None:
                t_add_prev_start = time.time()
                P_sum += previous_Runoff
                P5_sum += previous_Runoff
                t_add_prev_end = time.time()
                log_to_file(f"6. Add previous runoff: {t_add_prev_end - t_add_prev_start:.4f} seconds")

            t_m_start = time.time()
            m1_cp = compute_M(sr1, P5_sum)
            m2_cp = compute_M(sr2, P5_sum)
            m3_cp = compute_M(sr3, P5_sum)
            t_m_end = time.time()
            log_to_file(f"7. Compute M1,M2,M3: {t_m_end - t_m_start:.4f} seconds")

            t_runoff_start = time.time()
            R = calculate_runoff_cupy(P_sum, P5_sum, m1_cp, m2_cp, m3_cp, sr1, sr2, sr3)
            # R = runoff_total_volume(R)
            t_runoff_end = time.time()
            log_to_file(f"8. Calculate runoff: {t_runoff_end - t_runoff_start:.4f} seconds")

            del m1_cp, m2_cp, m3_cp

            t_transfer_start = time.time()
            previous_Runoff = transfer_flow(destination_index, R)
            t_transfer_end = time.time()
            log_to_file(f"9. Transfer flow: {t_transfer_end - t_transfer_start:.4f} seconds")

            t_extract_start = time.time()
            handler.save_tiff(cp.asnumpy(R), os.path.join(output_folder, f'runoff_simulation_{index}.tif'))
            t_extract_end = time.time()
            log_to_file(f"10. write runoff simulation result: {t_extract_end - t_extract_start:.4f} seconds")


            # Track value at coordinate without frequent CPU transfer
            # point_value += R[coordinate[1], coordinate[0]]
            # if (index + 1) % 8 == 0:
            #     t_extract_start = time.time()
            #     runoff_time_series.append(float(point_value.get()))
            #     point_value = cp.array(0.0)  # reset after saving
            #     t_extract_end = time.time()
            #     log_to_file(f"10. Extract point value: {float(point_value.get())} mm")
            #     log_to_file(f"10. Extract and reset point value: {t_extract_end - t_extract_start:.4f} seconds")

            del R

        iteration_end = time.time()
        total_time = iteration_end - iteration_start
        log_to_file(f"Total iteration time: {total_time:.4f} seconds")

    print("Processing complete!")
    log_to_file(f"Complete runoff simulation time: {time.time() - t_start_g:.4f} seconds")
    return runoff_time_series


def plot_runoff_time_series(runoff_values, output_path="runoff_time_series.png"):
    """
    Plot the runoff time series
    
    Args:
        runoff_values: List of runoff values (as CuPy objects)
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    # Convert CuPy values to NumPy/Python for plotting - all at once at the end
    values_numpy = [float(val) for val in runoff_values]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(values_numpy)), values_numpy, 'b-', linewidth=2, marker='o')
    plt.title('Runoff Time Series at Specified Coordinate')
    plt.xlabel('Time Step')
    plt.ylabel('Runoff Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# Example usage
if __name__ == "__main__":
    # folder_path = "rain_dataset_2023-07-01_2023-10-01" # lower ganga rain images path
    folder_path = "../tifs/masalia/masalia_daily_rain" # masalia rain images path
    output_folder_path = "../tifs/masalia/experiment_tifs/runoff_daily_simulation_2023_07-09" # masalia rain images path
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Define coordinate to extract (x, y)
    coordinate = (25944, 12661)
    
    # Process images and collect time series data
    runoff_series = process_images_and_track_coordinates(folder_path, coordinate, output_folder=output_folder_path)
    # Plot the results
    s_time = time.time()
    # plot_runoff_time_series(runoff_series)
    log_to_file(f"Time Series Time: {time.time() - s_time:.4f} seconds")

    print(f"Time series data for coordinate {coordinate} has been saved.")






# process_images_with_detailed_log('rain_dataset_2023-07-01_2023-10-01', 'runoff_simulation_tests_output')

# full_execution_end_time = time.time()


# print(f"Runoff Simulation overall time : {full_execution_end_time - full_execution_start_time}")