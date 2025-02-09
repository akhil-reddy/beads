"""
    Initialize the retina of the CMU. In this function, create the equivalent of a digital eye based on rods and cones'
    distribution. Study other computational models of the human eye to refine this data structure.
"""

import math
import random


class Cell:
    """
    A simple class to represent a retinal cell.

    Attributes:
        x (float): The x-coordinate of the cell.
        y (float): The y-coordinate of the cell.
        cell_type (str): The type of the cell ('rod' or 'cone').
    """

    def __init__(self, x, y, cell_type):
        self.x = x
        self.y = y
        self.cell_type = cell_type  # "rod" or "cone"

    def __repr__(self):
        return f"{self.cell_type.capitalize()}(x={self.x:.2f}, y={self.y:.2f})"


class Retina:
    """
    Represents a digital retina with a radial distribution of rods and cones.

    Attributes:
        cells (list of Cell): A list containing all the retinal cells.
    """

    def __init__(self, cells):
        self.cells = cells

    def __repr__(self):
        return f"Retina with {len(self.cells)} cells."


def initialize_retina(retina_radius=1.0, fovea_radius_ratio=0.3, n_rings=30):
    """
    Initializes a digital retina by distributing rods and cones on a circular manifold.

    Parameters:
        retina_radius (float): The maximum radius of the retina.
        fovea_radius_ratio (float): Ratio (0-1) that defines the fovea (central cone-dense area)
                                    relative to the overall retina radius.
        n_rings (int): Number of concentric rings to use for discretizing the retina.

    Returns:
        Retina: An instance of Retina with cells distributed in a radial pattern.
    """
    cells = []
    fovea_radius = retina_radius * fovea_radius_ratio

    # Loop over concentric rings from the center (r=0) to the outer edge.
    for i in range(n_rings + 1):
        # Compute the current radius for this ring.
        r = retina_radius * i / n_rings

        if i == 0:
            # At the very center, we add a single cell (always a cone).
            cells.append(Cell(0.0, 0.0, "cone"))
        else:
            # Determine the circumference of the current ring.
            circumference = 2 * math.pi * r
            # Define a desired spacing along the ring (this can be adjusted).
            spacing = retina_radius / n_rings
            # Calculate how many cells will fit along the circumference.
            n_cells_in_ring = max(1, int(circumference / spacing))

            for j in range(n_cells_in_ring):
                theta = 2 * math.pi * j / n_cells_in_ring  # Evenly spaced angles.
                x = r * math.cos(theta)
                y = r * math.sin(theta)

                # Decide on the cell type based on the radial distance.
                if r < fovea_radius:
                    # Inside the fovea: use cones.
                    cell_type = "cone"
                else:
                    # In the periphery: rods dominate.
                    # Here we use a probability: 90% rods, 10% cones.
                    cell_type = "cone" if random.random() < 0.1 else "rod"

                cells.append(Cell(x, y, cell_type))

    return Retina(cells)


# Example usage:
if __name__ == "__main__":
    retina = initialize_retina()
    print(retina)
    # Optionally, iterate over the cells to inspect their positions and types.
    for cell in retina.cells:
        print(cell)
