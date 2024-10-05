import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class PlotVisualizer:
    def __init__(self, figsize=(12, 6)):
        """
        Initializes the visualizer with a default figure size.
        """
        self.figsize = figsize

    def plot_with_scatter_missing_cbg(self, data,x_column, y_column,
                          title='Plot', xlabel='X-Axis', ylabel='Y-Axis', 
                          line_color='blue', scatter_color='red', scatter_marker='x', 
                          ylim=None, alpha=0.8, grid=True):
        """
        General function to plot data with a line and overlay scatter plot for missing data.

        Parameters:
        - data: DataFrame with main data to be plotted.
        - missing_data: DataFrame with missing data points (for scatter plot).
        - x_column: Column name for the X-axis (typically time or elapsed time).
        - y_column: Column name for the Y-axis (target variable to be plotted).
        - title: Title of the plot.
        - xlabel: X-axis label.
        - ylabel: Y-axis label.
        - line_color: Color of the line plot.
        - scatter_color: Color of the scatter plot for missing data.
        - scatter_marker: Marker style for missing data points.
        - ylim: Tuple (ymin, ymax) to define the limits of the Y-axis.
        - alpha: Transparency level of the line plot.
        - grid: Boolean to show grid on the plot.
        """
        missing_data = data[data["missing_cbg"] == 1]

        # Create the figure with specified size
        plt.figure(figsize=self.figsize)

        # Plot the main data as a line plot
        plt.plot(data[x_column], data[y_column], color=line_color, label=f'missing_cbg (Present)', alpha=alpha)

        # Scatter plot for missing data (use np.full to set a constant y-value like 0)
        plt.scatter(missing_data[x_column], np.full(len(missing_data), 0), 
                    color=scatter_color, label='Missing Data (Gaps)', marker=scatter_marker)

        # Set title, labels, and limits
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Optionally set y-axis limits
        if ylim is not None:
            plt.ylim(ylim)

        # Show the legend and grid
        plt.legend()
        plt.grid(grid)

        # Optimize layout
        plt.tight_layout()

        # Display the plot
        plt.show()

    def plot_corr_matrix_upper_triangle(self, corr_matrix, cmap='coolwarm', linewidths=0.5, title="Correlation Map"):
        """
        Plots the upper triangle of a correlation matrix as a heatmap.

        Parameters:
        - corr_matrix: The correlation matrix (typically a Pandas DataFrame).
        - cmap: Colormap for the heatmap.
        - linewidths: Line width for heatmap grid lines.
        - title: Title of the heatmap.
        """
        plt.figure(figsize=(10, 8))

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot the heatmap with the mask
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, linewidths=linewidths)

        # Set the title and show the plot
        plt.title(title)
        plt.show()