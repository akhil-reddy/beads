"""
    Initialize the retina of the CMU. In this function, create the equivalent of a digital eye based on rods and cones'
    distribution. Study other computational models of the human eye to refine this data structure.
"""

import math


class Cone:
    """
        The Cones in the retina should exhibit three main properties:
        1. They should have a high activation threshold and hence avoid noisy stray photons (many false negatives)
        2. They should map almost one-to-one onto bipolar cells i.e., Push implementation. This is done via having
        high sensitivity to different combinations of colours. Imagine a friend who considers red and crimson the same
        colour and someone with better color sensitivity. This difference is primarily due to cone stimulus registration
        3. They SHOULD have faster activation and deactivation kinetics
    """

    def __init__(self):
        pass


class Rod:
    """
        The Rods in the retina should exhibit three main properties:
        1. They should have a low activation threshold (many false positives)
        2. They should map many-to-one onto bipolar cells i.e., Push implementation. This is done by having low
        sensitivity to different colours with flatter overlapping distributions (negative kurtosis)
        3. They SHOULD have slower activation and deactivation kinetics. This slower response aids in integrating photon
         signals over time, boosting sensitivity but sacrificing temporal resolution. In the biological retina, this is
         done to ensure that the single photon from the scenery picked up by Rh* is not noise / stray photons
         from the environment. HOWEVER, as modern digital sensor already incorporate this concept for low light vision,
         we don't need to implement it yet
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

    def __init__(self, x, y, cell_type, shape, subtype=None, activation_threshold=None):
        self.x = x
        self.y = y
        self.cell_type = cell_type  # "cone" or "rod"
        self.shape = shape  # "triangle" or "hexagon"
        self.subtype = subtype
        self.activation_threshold = activation_threshold

    def __repr__(self):
        return f"{self.cell_type.capitalize()} ({self.shape}) at (x={self.x:.2f}, y={self.y:.2f})"


class Retina:
    """
        Represents a digital retina based on a 2D hexagonal surface.

        Attributes:
            cells (list of Cell): All cells in the retina.
            surface_radius (float): The radius of the circular area covered.
            cone_threshold (float): Hexagons with centers closer than this will be subdivided.
    """

    def __init__(self, cells, surface_radius, cone_threshold):
        self.cells = cells
        self.surface_radius = surface_radius
        self.cone_threshold = cone_threshold

    def __repr__(self):
        return (f"Retina with {len(self.cells)} cells "
                f"(surface radius: {self.surface_radius}, cone threshold: {self.cone_threshold}).")


# Ratios are consistent with human eye geometry
def initialize_retina(surface_radius=1248.0, cone_threshold=208.0, hex_size=1.0):
    """
    Initializes a digital retina by distributing rods and cones on a circular (radial) manifold.

    The fovea (central region) is defined as the area with radius = retina_radius * fovea_radius_ratio.
    In the fovea, every cell is a cone (100% probability), whereas in the periphery the probability
    of a cell being a cone decreases linearly to 'cone_prob_edge' at the retina edge.
    Additionally, if a cell is a cone, its subtype (S, M, or L) is selected according to a specified
    probability distribution.

    Parameters:
        surface_radius (float): Radius of the circular area.
        cone_threshold (float): Distance threshold for subdivision.
        hex_size (float): The "radius" of each hexagon (distance from center to a vertex).

    Returns:
        Retina: An instance containing the generated cells.
    """
    cells = []
    # For pointy-topped hexagons:
    hex_width = math.sqrt(3) * hex_size
    hex_height = 2 * hex_size
    vertical_spacing = 0.75 * hex_height  # vertical distance between rows

    # Determine row and column ranges to cover the area.
    row_min = int(math.floor(-surface_radius / vertical_spacing))
    row_max = int(math.ceil(surface_radius / vertical_spacing))
    col_min = int(math.floor(-surface_radius / hex_width))
    col_max = int(math.ceil(surface_radius / hex_width))

    for row in range(row_min, row_max + 1):
        y = row * vertical_spacing
        # Offset every other row for hexagonal tiling.
        offset = hex_width / 2 if row % 2 != 0 else 0
        for col in range(col_min, col_max + 1):
            x = offset + col * hex_width
            # Only include hexagon centers within the circular surface.
            if math.sqrt(x * x + y * y) <= surface_radius:
                distance = math.sqrt(x * x + y * y)
                if distance < cone_threshold:
                    # Subdivide the hexagon into 6 triangles.
                    # First, compute the 6 vertices of the hexagon.
                    vertices = []
                    for k in range(6):
                        angle = math.pi / 3 * k
                        vx = x + hex_size * math.cos(angle)
                        vy = y + hex_size * math.sin(angle)
                        vertices.append((vx, vy))
                    # Create 6 triangular cells by taking the center and each pair of adjacent vertices.
                    for k in range(6):
                        v1 = vertices[k]
                        v2 = vertices[(k + 1) % 6]
                        # Compute the centroid of the triangle (for illustrative positioning).
                        cx = (x + v1[0] + v2[0]) / 3.0
                        cy = (y + v1[1] + v2[1]) / 3.0
                        cells.append(Cell(cx, cy, cell_type="cone", shape="triangle"))
                else:
                    # Outside the fovea: divide the hexagon into 3 equal parallelograms.
                    # For a pointy-topped hexagon of "radius" hex_size,
                    # the distance between parallel sides is hex_size * sqrt(3).
                    # One-third of that distance is: hex_size / sqrt(3).
                    # We choose the subdivision direction along 30° (unit vector u = (cos30, sin30)).
                    u_x = math.cos(math.radians(30))  # √3/2
                    u_y = math.sin(math.radians(30))  # 1/2
                    offset_distance = hex_size / math.sqrt(3)
                    for factor in [-1, 0, 1]:
                        cx = x + factor * offset_distance * u_x
                        cy = y + factor * offset_distance * u_y
                        cells.append(Cell(cx, cy, cell_type="rod", shape="parallelogram"))

    return Retina(cells, surface_radius, cone_threshold)


if __name__ == "__main__":
    retina = initialize_retina(surface_radius=1.0, cone_threshold=0.3, hex_size=0.1)
    print(retina)
    for cell in retina.cells:
        print(cell)

    # Initialize the Cochlea
