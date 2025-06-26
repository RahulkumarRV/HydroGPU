# HydroGPU

HydroGPU is a high-performance hydrological analysis toolkit that leverages GPU acceleration (via CuPy) for large-scale raster processing. It provides tools for downloading rainfall and geospatial data from Google Earth Engine (GEE), and implements efficient algorithms for flow accumulation, stream order, runoff simulation, and micro-watershed delineation.

---

## Workflow Overview

1. **Download Data from GEE to Local Storage**
2. **Run Hydrological Algorithms (Flow Accumulation, Stream Order, Runoff Simulation, Micro-watershed Delineation)**

---

## 1. Downloading Data from Google Earth Engine

The first step is to download rainfall or other geospatial datasets from GEE to your local machine. This is handled by the `gee_processing.py` module.

### Steps:

- **Configure your parameters** (project ID, dataset ID, asset ID, date range, etc.) in your script or directly in `gee_processing.py`.
- **Run the GEE data downloader** to export and download the data from your GEE account to Google Drive, and then to your local folder.

#### Example Usage

```python
from gee_processing import GEEDataDownloader

project_id = 'your-gcp-project-id'
dataset_id = 'JAXA/GPM_L3/GSMaP/v6/operational'
asset_id = 'projects/your-gcp-project/assets/your_geometry'
start_date = '2023-07-01'
end_date = '2023-07-10'
drive_folder = 'Rainfall_Export'
local_folder = 'downloaded_GEE_rain'

downloader = GEEDataDownloader(project_id)
downloader.execute_gee_tasks(dataset_id, asset_id, start_date, end_date, drive_folder, local_folder)
```

- This will download and merge the rainfall images into your specified local folder.

---

## 2. Running Hydrological Algorithms

Once your data is available locally (GeoTIFFs), you can run any of the following algorithms. Each algorithm requires specific input files (usually GeoTIFF rasters) and produces output files (typically GeoTIFFs or PNGs) in your chosen directory.

### A. Flow Accumulation

- **File:** `flow_accumulation.py`
- **Inputs:**  
  - Flow direction raster (GeoTIFF, D8 encoding, e.g. from DEM processing)
- **Outputs:**  
  - Flow accumulation raster (GeoTIFF; each cell shows the number of upstream cells draining through it)
- **How to Run:**  
  - Edit the input file path for the flow direction raster at the top of `flow_accumulation.py`.
  - Run the script. The output flow accumulation raster will be saved to the specified location.

### B. Stream Order (Strahler or Shreve)

- **Files:** `Strahler_so.py`, `Shreve_so.py`
- **Inputs:**  
  - Flow direction raster (GeoTIFF, D8 encoding)
  - Flow accumulation raster (GeoTIFF)
  - Stream threshold value (integer; minimum accumulation to define a stream)
- **Outputs:**  
  - Stream order raster (GeoTIFF; each cell labeled with its stream order)
- **How to Run:**  
  - Set the input file paths for flow direction and accumulation, and adjust the threshold in the script.
  - Run the script. The output stream order raster will be saved to the specified location.

### C. Runoff Simulation

- **File:** `runoff_simulation.py`
- **Inputs:**  
  - Rainfall raster(s) (GeoTIFF; typically merged 3-hour or daily rainfall)
  - Flow direction raster (GeoTIFF)
  - DEM raster (GeoTIFF)
  - (Optional) Soil, land use, or other parameter rasters
- **Outputs:**  
  - Runoff raster(s) (GeoTIFF; simulated runoff for each cell)
  - (Optional) Plots or summary statistics
- **How to Run:**  
  - Configure the input dataset paths and output folder in the script.
  - Run the script. The output runoff rasters will be saved to the specified location.

### D. Micro-watershed Delineation

- **File:** `microwatershed.py`
- **Inputs:**  
  - Flow direction raster (GeoTIFF)
  - Stream segment raster or vector (GeoTIFF or shapefile; defines stream network)
  - (Optional) Flow accumulation raster (GeoTIFF)
- **Outputs:**  
  - Micro-watershed raster (GeoTIFF; each cell labeled with its micro-watershed ID)
- **How to Run:**  
  - Set the input file paths for flow direction and stream segment data in the script.
  - Run the script. The output micro-watershed raster will be saved to the specified
  
---

## Notes

- All scripts are designed to run efficiently on GPU using CuPy. Make sure you have a compatible GPU and the necessary drivers installed.
- Input and output file paths should be updated as per your data locations.
- For custom workflows, you can import and use the functions/classes from each script in your own Python code.

---

## Requirements

- Python 3.8+
- CUDA version 11+
- Cucim
- [CuPy](https://cupy.dev/) (for GPU acceleration)
- [NumPy](https://numpy.org/)
- [rasterio](https://rasterio.readthedocs.io/)
- [PyDrive](https://pythonhosted.org/PyDrive/)
- [Google Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- tqdm, matplotlib (for progress bars and plotting, optional)

---

## Example Workflow

1. **Download rainfall data from GEE:**
    - Use `gee_processing.py` to export and download rainfall images.

2. **Run flow accumulation:**
    - Use `flow_accumulation.py` to compute the flow accumulation raster.

3. **Calculate stream order:**
    - Use `Strahler_so.py` or `Shreve_so.py` for stream order calculation.

4. **Delineate micro-watersheds:**
    - Use `microwatershed.py` to generate micro-watershed boundaries.


5. **Simulate runoff:**
    - Use `runoff_simulation.py` to process rainfall and compute runoff.

---

## License

This project is for research and educational purposes. Please cite appropriately if used in publications.

---