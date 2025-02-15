"""
    Initialize the retina of the CMU. In this function, create the equivalent of a digital eye based on rods and cones'
    distribution. Study other computational models of the human eye to refine this data structure.
"""

import math
import random


class Cone:
    """
        In the fovea, make sure to use principles from loop subdivision to enrich the resolution of images as follows:
            1. Let's suppose there are three photons carrying different colours A, B and C
            2. If there exists only two cones to absorb those three photos, usually there's a merge between photon A
                & B, and B & C
            3. To extract the original tri-colours out, need to find B. However there is an assumption here that there
            exists enough of A & C to individually remove them from the boundary colour
    """
    def __init__(self):
        pass

class Cell:
    """
    Represents a retinal cell.

    Attributes:
        x (float): The x-coordinate of the cell.
        y (float): The y-coordinate of the cell.
        cell_type (str): The type of the cell ('rod' or 'cone').
        subtype (str or None): For cone cells, one of 'S', 'M', or 'L'. For rods, this is None.
        activation_threshold: A minimum brightness level for that cell to pickup colour
    """

    def __init__(self, x, y, cell_type, subtype=None, activation_threshold=None):
        self.x = x
        self.y = y
        self.cell_type = cell_type  # "rod" or "cone"
        self.subtype = subtype  # For cones: "S", "M", or "L"
        self.activation_threshold = activation_threshold  # rods have a low threshold and cones have a high threshold

    def __repr__(self):
        if self.cell_type == "cone" and self.subtype:
            return f"Cone({self.subtype}) at (x={self.x:.2f}, y={self.y:.2f})"
        else:
            return f"{self.cell_type.capitalize()} at (x={self.x:.2f}, y={self.y:.2f})"


class Retina:
    """
    Represents a digital retina with a radial distribution of rods and cones.

    Attributes:
        cells (list of Cell): A list of all retinal cells.
        retina_radius (float): The maximum radius of the retina.
        fovea_radius (float): The radius of the cone-dense foveal region.
    """

    def __init__(self, cells, retina_radius, fovea_radius):
        self.cells = cells
        self.retina_radius = retina_radius
        self.fovea_radius = fovea_radius

    def __repr__(self):
        return (f"Retina with {len(self.cells)} cells "
                f"(retina radius: {self.retina_radius}, fovea radius: {self.fovea_radius}).")


def initialize_retina(retina_radius=1.0, fovea_radius_ratio=0.3, n_rings=30,
                      cone_prob_fovea=1.0, cone_prob_edge=0.05,
                      cone_subtype_distribution=None):
    """
    Initializes a digital retina by distributing rods and cones on a circular (radial) manifold.

    The fovea (central region) is defined as the area with radius = retina_radius * fovea_radius_ratio.
    In the fovea, every cell is a cone (100% probability), whereas in the periphery the probability
    of a cell being a cone decreases linearly to 'cone_prob_edge' at the retina edge.
    Additionally, if a cell is a cone, its subtype (S, M, or L) is selected according to a specified
    probability distribution.

    Parameters:
        retina_radius (float): The overall radius of the retina.
        fovea_radius_ratio (float): The ratio (0-1) defining the fovea relative to retina_radius.
        n_rings (int): Number of concentric rings used for discretizing the retina.
        cone_prob_fovea (float): The probability a cell is a cone in the fovea (typically 1.0).
        cone_prob_edge (float): The probability a cell is a cone at the retinal periphery.
        cone_subtype_distribution (dict): A dict with keys 'S', 'M', 'L' and their associated probabilities.

    Returns:
        Retina: An instance of Retina with cells distributed in a radial pattern.
    """
    if cone_subtype_distribution is None:
        cone_subtype_distribution = {'S': 0.1, 'M': 0.45, 'L': 0.45}
    cells = []
    fovea_radius = retina_radius * fovea_radius_ratio

    # Define a function to compute the probability that a cell is a cone based on its radial distance.
    def cone_probability(r):
        if r <= fovea_radius:
            return cone_prob_fovea
        else:
            # Linearly interpolate from cone_prob_fovea at the fovea edge to cone_prob_edge at the retina edge.
            return cone_prob_fovea + (cone_prob_edge - cone_prob_fovea) * (
                        (r - fovea_radius) / (retina_radius - fovea_radius))

    # Loop over concentric rings from the center (r = 0) to the outer edge (r = retina_radius).
    for i in range(n_rings + 1):
        # Compute the current radius for this ring.
        r = retina_radius * i / n_rings

        if i == 0:
            # At the very center, add a single cone cell.
            subtype = random.choices(
                list(cone_subtype_distribution.keys()),
                weights=list(cone_subtype_distribution.values())
            )[0]
            cells.append(Cell(0.0, 0.0, "cone", subtype))
        else:
            # Determine the number of cells along this ring based on its circumference.
            circumference = 2 * math.pi * r
            # Use a spacing factor (can be tuned) to determine how many cells to place.
            spacing = retina_radius / n_rings
            n_cells_in_ring = max(1, int(circumference / spacing))

            for j in range(n_cells_in_ring):
                theta = 2 * math.pi * j / n_cells_in_ring  # Evenly spaced angles.
                x = r * math.cos(theta)
                y = r * math.sin(theta)

                # Decide cell type based on radial distance.
                prob = cone_probability(r)
                if random.random() < prob:
                    cell_type = "cone"
                    # Choose cone subtype according to the distribution.
                    subtype = random.choices(
                        list(cone_subtype_distribution.keys()),
                        weights=list(cone_subtype_distribution.values())
                    )[0]
                    cells.append(Cell(x, y, cell_type, subtype))
                else:
                    cell_type = "rod"
                    cells.append(Cell(x, y, cell_type))

    return Retina(cells, retina_radius, fovea_radius)


# Example usage:
if __name__ == "__main__":
    retina = initialize_retina()
    print(retina)
    # Optionally, inspect each cell.
    for cell in retina.cells:
        print(cell)
