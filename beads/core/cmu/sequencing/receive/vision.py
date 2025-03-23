"""
    Initialize the retina of the CMU. In this function, create the equivalent of a digital eye based on rods and cones'
    distribution. Study other computational models of the human eye to refine this data structure.
"""

import math
import random
import numpy as np

from beads.core.cmu import Retina


# -- Govardovskii nomogram functions ------------------------------------------
def govardovskii_nomogram(wavelength, lambda_max):
    """
    Calculates the relative spectral sensitivity using a Govardovskii nomogram template.
    (This is a simplified version.)

    Args:
        wavelength (numpy.ndarray or float): Wavelength(s) in nanometers.
        lambda_max (float): The peak absorbance wavelength (nm) for the pigment.

    Returns:
         numpy.ndarray or float: Relative sensitivity.
    """
    k = 69.7
    return np.exp(3.21 * np.log(k / wavelength)
                  - 0.485 * (np.log(k / wavelength)) ** 2
                  + 9.71e-3 * (np.log(k / wavelength)) ** 3)


def spectral_sensitivity(wavelength, lambda_max):
    """
    Shifts the Govardovskii nomogram so that the peak occurs at lambda_max.

    Args:
        wavelength (float or np.ndarray): Wavelength(s) in nm.
        lambda_max (float): Peak sensitivity wavelength in nm.

    Returns:
        float or np.ndarray: Relative sensitivity.
    """
    # This simple approach scales the input wavelength so that when lambda_max==500 nm, no shift occurs.
    effective_wavelength = wavelength * (500.0 / lambda_max)
    return govardovskii_nomogram(effective_wavelength, lambda_max)


# -- HSP Brightness (from alienryderflex.com) -------------------------------
def hsp_brightness(R, G, B):
    """
    Calculate perceived brightness using the HSP model.

    Args:
        R, G, B (float): Red, Green, Blue components (0-255).

    Returns:
        float: Perceived brightness.
    """
    return math.sqrt(0.299 * (R ** 2) + 0.587 * (G ** 2) + 0.114 * (B ** 2))


# -- Conversion from RGB to Luminance ----------------------------------------
def rgb_to_luminance(R, G, B, L_max=300, gamma=2.2):
    """
    Converts an RGB triplet into an estimated luminance (cd/m²) using the HSP model.
    Assumes that (255,255,255) corresponds to L_max cd/m².

    Args:
        R (int): RGB values (0-255).
        G (int): RGB values (0-255).
        B (int): RGB values (0-255).
        L_max (float): Maximum luminance (cd/m²) for white.
        gamma (float): Display gamma.

    Returns:
        float: Estimated luminance in cd/m².
    """
    # Compute perceived brightness (HSP).
    perceived = hsp_brightness(R, G, B)
    # Maximum perceived brightness for (255,255,255)
    max_perceived = hsp_brightness(255, 255, 255)
    # Normalize to [0, 1]
    norm = perceived / max_perceived
    # Apply gamma correction and scale to L_max.
    return (norm ** gamma) * L_max


def luminance_to_photoisomerizations(luminance, pupil_diameter_mm=3.0,
                                     conversion_factor=1.0, ocular_transmittance=0.9):
    """
    Estimate the number of photoisomerizations per receptor per second from luminance.

    Args:
        luminance (float): Luminance in cd/m².
        pupil_diameter_mm (float): Pupil diameter in mm (assumed circular).
        conversion_factor (float): Photoisomerizations per receptor per troland per second.
        ocular_transmittance (float): Fraction of light transmitted (default 0.9).

    Returns:
        float: Photoisomerizations per receptor per second.
    """
    pupil_area = math.pi * (pupil_diameter_mm / 2.0) ** 2  # in mm²
    # 1 troland = 1 cd/m²·mm²
    trolands = luminance * pupil_area
    return conversion_factor * trolands * ocular_transmittance


# -- Photoreceptor Classes ----------------------------------------------------

class Cone:
    """
    Cone photoreceptors for color vision:
      1. High activation threshold (around 100 photon–equivalents) so that low-intensity noise (many false negatives)
      is ignored
      2. They have a nearly one-to-one mapping onto bipolar cells i.e., Push implementation
      3. They show distinct spectral sensitivity (using a narrow Govardovskii nomogram). Imagine a friend who considers
      red and crimson the same colour and someone with better color sensitivity. This difference is primarily
      due to cone stimulus registration
      4. They SHOULD have faster activation and deactivation kinetics


    Attributes:
        threshold (float): Minimum brightness required for activation.
        subtype (str): 'S', 'M', or 'L'
    """

    def __init__(self, subtype=None, threshold=100):
        self.threshold = threshold
        self.subtype = (subtype if subtype in ['S', 'M', 'L']
                        else random.choices(['S', 'M', 'L'], weights=[0.1, 0.45, 0.45])[0])
        if self.subtype == 'S':
            self.lambda_max = 445  # nm
        elif self.subtype == 'M':
            self.lambda_max = 535  # nm
        elif self.subtype == 'L':
            self.lambda_max = 565  # nm

    """
    Process an RGB pixel given a dominant wavelength (nm) of the stimulus.
    Only processes if the estimated brightness exceeds the cone threshold.

    Args:
        R, G, B (int): RGB values (0-255).
        wavelength (float): Dominant wavelength (nm).

    Returns:
        float: Response magnitude (photoisomerizations per receptor per second).
    """

    def process_pixel(self, R, G, B, wavelength):

        # Convert RGB to luminance (using default calibration assumptions)
        luminance = rgb_to_luminance(R, G, B)
        # Estimate photoisomerizations (using default conversion factor for cones)
        photoisom = luminance_to_photoisomerizations(luminance, conversion_factor=100.0)
        # Apply cone threshold: if below threshold, response is 0.
        if photoisom < self.threshold:
            return 0.0
        # Weight by spectral sensitivity using the Govardovskii nomogram.
        spectral_weight = spectral_sensitivity(wavelength, self.lambda_max)
        return (photoisom - self.threshold) * spectral_weight

    """
    The signals from the cones are then combined to produce opponent channels for Push implementation

    Args:
        L, M, S (int): The wavelength equivalents of RGB

    Returns:
        float: Red-Green, Blue-Yellow and Luminance channels
    """

    @staticmethod
    def get_opponent_channels(L, M, S):
        # Calculate opponent channels
        rg = L - M
        by = S - 0.5 * (L + M)
        lum = L + M
        return rg, by, lum

    def __repr__(self):
        return f"Cone({self.subtype}) [λ_max={self.lambda_max}nm, threshold={self.threshold}]"


class Rod:
    """
    Rod photoreceptors for low-light vision:
      1. Very low activation threshold (near 1 photon–equivalent) so that even dim light is detected (many false
      positives)
      2. They converge many-to-one onto bipolar cells i.e., Push implementation
      3. They use a broader Govardovskii nomogram (flatter, overlapping curve) for spectral sensitivity.
      4. They integrate signal over multiple iterations (to reduce false positives from noise). They SHOULD have
        slower activation and deactivation kinetics. This slower response aids in integrating photon
        signals over time, boosting sensitivity but sacrificing temporal resolution. In the biological retina, this is
        done to ensure that the single photon from the scenery picked up by Rh* is not noise / stray photons
        from the environment. HOWEVER, as modern digital sensor already incorporate this concept for low light vision,
        we don't need to implement it yet

    Attributes:
        threshold (float): Minimal brightness needed (set to 1 photon–equivalent).
        n_iterations (int): Number of iterations for signal integration.
    """

    def __init__(self, threshold=1, n_iterations=10):
        self.threshold = threshold
        self.lambda_max = 498  # nm typical for rods.
        self.n_iterations = n_iterations
        self.responses = []

    """
    Process an RGB pixel for a rod by integrating over multiple iterations.

    Args:
        R, G, B (int): RGB values (0-255).
        wavelength (float): Dominant wavelength (nm).

    Returns:
        float: Averaged response (photoisomerizations per receptor per second).
    """

    def process_pixel(self, R, G, B, wavelength):
        luminance = rgb_to_luminance(R, G, B)
        photoisom = luminance_to_photoisomerizations(luminance,
                                                     conversion_factor=1.0)
        if len(self.responses) >= self.n_iterations:
            self.responses.pop()

        if photoisom < self.threshold:
            self.responses.append(0.0)
        else:
            spectral_weight = spectral_sensitivity(wavelength, self.lambda_max)
            self.responses.append((photoisom - self.threshold) * spectral_weight)

        return sum(self.responses) / self.n_iterations

    def __repr__(self):
        return f"Rod [λ_max={self.lambda_max}nm, threshold={self.threshold}, iterations={self.n_iterations}]"


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


def create_cones(x, y, hex_size):
    vertices = []
    cells = []
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

    return cells


# Ratios are consistent with human eye geometry
def initialize_photoreceptors(retina, surface_radius=1248.0, cone_threshold=208.0, hex_size=1.0):  # radius in microns
    """
    Initializes photoreceptor part of the retina by distributing rods and cones on a circular (radial) manifold.

    The fovea (central region) is defined as the area with radius = retina_radius * fovea_radius_ratio.
    In the fovea, every cell is a cone (100% probability), whereas in the periphery the probability
    of a cell being a cone decreases linearly to 'cone_prob_edge' at the retina edge.
    Additionally, if a cell is a cone, its subtype (S, M, or L) is selected according to a specified
    probability distribution.

    Parameters:
        retina (object): the umbrella object holding the retina instance
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
                    cells.append(create_cones(x, y, hex_size))
                else:
                    # Compute a quadratic probability for rods that peaks at the midpoint.
                    # The peak is at r_peak, which is the midpoint between cone_threshold and surface_radius.
                    r_peak = (cone_threshold + surface_radius) / 2.0
                    # Quadratic function: probability is 0 at distance = cone_threshold and distance =
                    # surface_radius, and equals 1 at distance = r_peak. Choose exponents: a higher exponent for the
                    # inner side (steeper) and a lower one for the outer side (more gradual).
                    inner_exponent = 2.0  # steeper rise from fovea edge to r_peak
                    outer_exponent = 0.3  # more gradual fall-off from r_peak to the retina edge
                    if distance <= r_peak:
                        # Inner side: steep increase.
                        rod_probability = ((distance - cone_threshold) / (r_peak - cone_threshold)) ** inner_exponent
                    else:
                        # Outer side: more gradual decrease.
                        rod_probability = ((surface_radius - distance) / (surface_radius - r_peak)) ** outer_exponent
                    # Clamp to ensure the probability is between 0 and 1.
                    rod_probability = max(0.0, min(rod_probability, 1.0))

                    for factor in [-1, 0, 1]:
                        # Only create a rod if a random check passes, with probability based on distance.
                        if random.random() < rod_probability:
                            # Outside the fovea: divide the hexagon into 3 equal parallelograms.
                            # For a pointy-topped hexagon of "radius" hex_size, the distance between parallel sides is
                            # hex_size * sqrt(3). One-third of that distance is: hex_size / sqrt(3).
                            # We choose the subdivision direction along 30° (unit vector u = (cos30, sin30)).
                            u_x = math.cos(math.radians(30))  # √3/2
                            u_y = math.sin(math.radians(30))  # 1/2
                            offset_distance = hex_size / math.sqrt(3)

                            cx = x + factor * offset_distance * u_x
                            cy = y + factor * offset_distance * u_y
                            cells.append(Cell(cx, cy, cell_type="rod", shape="parallelogram"))
                        # Create cones near to the fovea as well
                        elif distance < r_peak:
                            cells.append(create_cones(x, y, hex_size))

    retina.init_photoreceptors(cells, surface_radius, cone_threshold)
    return retina
