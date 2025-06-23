import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import rasterio

def extract_index(filename):
    """Extract the index number from the filename."""
    match = re.search(r'runoff_simulation_3H_(\d+)\.tif', filename)
    if match:
        return int(match.group(1))
    return 0

def process_tif_files(folder_path, coordinate=(100, 100)):
    """
    Process TIF files in the specified folder:
    1. Load and sort TIF files by index
    2. Group into batches of 14
    3. Sum each batch and extract value at the specified coordinate
    4. Return time series data
    
    Args:
        folder_path: Path to folder containing TIF files
        coordinate: (x, y) coordinate to extract values from
        
    Returns:
        time_series: List of values at the specified coordinate for each batch
    """
    # Get all TIF files in the folder
    tif_files = glob(os.path.join(folder_path, "runoff_simulation_3H_*.tif"))
    
    # Sort files by their index
    tif_files.sort(key=extract_index)
    
    # Group files into batches of 14
    batches = [tif_files[i:i+14] for i in range(0, len(tif_files), 14)]
    
    time_series = []
    
    # Process each batch
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx+1}/{len(batches)}...")
        
        # Initialize sum array for the first file in batch
        with rasterio.open(batch[0]) as src:
            batch_sum = src.read(1).astype(np.float32)
            
        # Add the rest of the files in the batch
        for file_path in batch[1:]:
            with rasterio.open(file_path) as src:
                batch_sum += src.read(1).astype(np.float32)
        
        # Extract value at the specified coordinate
        value_at_coordinate = batch_sum[coordinate[1], coordinate[0]]
        time_series.append(value_at_coordinate)
        
    return time_series

def plot_time_series(time_series):
    """Plot the time series data."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(time_series) + 1), time_series, 'o-', linewidth=2)
    plt.title('Runoff Simulation Time Series')
    plt.xlabel('Batch Number (each batch = sum of 14 images)')
    plt.ylabel('Value at Specified Coordinate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('runoff_simulation_14_day_timeseries.png')
    plt.show()

if __name__ == "__main__":
    # Set the folder path containing the TIF files
    folder_path = "./runoff_simulation"
    
    # Set the coordinate to extract values from (x, y)
    # Note: You should replace this with your desired coordinate
    coordinate = (21074, 7667)
    
    print(f"Processing TIF files from {folder_path}")
    print(f"Extracting values at coordinate {coordinate}")
    
    # Process the TIF files and get the time series
    time_series = process_tif_files(folder_path, coordinate)
    
    print("Time series values:")
    for i, value in enumerate(time_series):
        print(f"Batch {i+1}: {value}")
    
    # Plot the time series
    plot_time_series(time_series)