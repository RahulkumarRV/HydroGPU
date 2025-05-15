import rasterio
import numpy as np

class GeoTIFFHandler:
    def __init__(self, tiff_path):
        """
        Initialize by loading an existing GeoTIFF file and storing its properties.
        """
        with rasterio.open(tiff_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.width = src.width
            self.height = src.height
            self.dtype = src.dtypes[0]  # Get the data type of the first band
            self.count = src.count  # Number of bands
            
            # Read original data (optional)
            # self.original_data = src.read(1)
            

        print(f"Loaded TIFF: {tiff_path}")
        print(f"CRS: {self.crs}, Transform: {self.transform}, Size: {self.width}x{self.height}, Type: {self.dtype}")

    def save_tiff(self, new_data, output_path, compression="LZW"):
        """
        Save a new TIFF file using the stored properties but with new data.

        :param new_data: 2D NumPy array containing new raster data.
        :param output_path: Path to save the new TIFF file.
        :param compression: Compression type for the TIFF file (default: "LZW").
        """
        if new_data.shape != (self.height, self.width):
            raise ValueError(f"Data shape {new_data.shape} does not match the stored shape ({self.height}, {self.width})")

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=self.count,
            dtype=new_data.dtype,
            crs=self.crs,
            transform=self.transform,
            compress=compression
        ) as dst:
            dst.write(new_data, 1)  # Write new data to band 1

        print(f"Saved new TIFF with compression='{compression}': {output_path}")

# Example Usage
# handler = GeoTIFFHandler("input.tif")  # Load the properties from an existing file

# # Create some new data (example: all zeros)
# new_data = np.zeros((handler.height, handler.width), dtype=handler.dtype)

# handler.save_tiff(new_data, "output.tif")  # Save new data with the same properties
