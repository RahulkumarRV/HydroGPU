import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_plot(data, output_path, colors=['brown', 'blue', 'green', 'yellow', 'orange', 'red'], normalize=False, min=0, max=1):
    """
    Creates and saves a plot of the given data using a custom colormap.

    Args:
        data (array-like): The data to visualize.
        output_path (str): Full path where the output image should be saved (including filename).
        colors (list): A list of colors to use in the colormap.
    """
    # Create a custom colormap
        
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # Plot the data
    if normalize:
        plt.imshow(data, cmap=cmap, vmin=min, vmax=max)
    else:
        plt.imshow(data, cmap=cmap)
    plt.colorbar()

    # Save the plot to the specified path
    plt.savefig(output_path)
    plt.clf()
