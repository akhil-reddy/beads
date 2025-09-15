from beads.core.cmu.sequencing.combine.vision import *
from beads.core.cmu.sequencing.receive.vision import *
from beads.core.cmu.sequencing.transforms.vision import *
from beads.core.cmu.transportation.vision import *


class Retina:
    """
        Represents a digital retina based on a 2D hexagonal surface.
    """

    def __init__(self):
        self.surface_radius = None
        self.cone_threshold = None

        self.photoreceptor_cells = None

        self.horizontal_cells = None

        self.rod_bipolar_cells = None
        self.cone_bipolar_cells = None

        self.aii_amacrine_cells = None
        self.star_amacrine_cells = None

        self.dsgc = None
        self.midget_ganglion_cells = None
        self.parasol_ganglion_cells = None
        self.small_bistrat_ganglion_cells = None

    """
    Function initalizes the photoreceptors and organizes them.
    """
    def init_photoreceptors(self, surface_radius, cone_threshold):
        self.surface_radius = surface_radius
        self.cone_threshold = cone_threshold

        self.photoreceptor_cells = initialize_photoreceptors()

    """
    Function initalizes the horizontal cells and organizes them.
    """
    def init_horizontal_cells(self):
        self.horizontal_cells = initialize_horizontal_cells(self.photoreceptor_cells)

    """
    Function initalizes the rod bipolar cells and organizes them.
    """
    def init_rod_bipolar_cells(self):
        self.rod_bipolar_cells = initialize_rod_bipolar_cells(self.photoreceptor_cells)

    """
    Function initalizes the AII amacrine cells and organizes them.
    """
    def init_aii_amacrine_cells(self):
        self.aii_amacrine_cells = initialize_aii_amacrine_cells(self.rod_bipolar_cells)

    """
    Function initalizes the cone bipolar cells and organizes them.
    """
    def init_cone_bipolar_cells(self):
        self.cone_bipolar_cells = initialize_cone_bipolar_cells(self.horizontal_cells, self.aii_amacrine_cells)

    """
    Function initalizes the star amacrine cells and organizes them.
    """
    def init_star_amacrine_cells(self):
        self.star_amacrine_cells = initialize_starburst_amacrine_cells(self.cone_bipolar_cells)

    """
    Function initalizes the ganglion cells and organizes them.
    """
    def init_ganglion_cells(self):
        self.dsgc = initialize_DSGCs(self.star_amacrine_cells)
        self.midget_ganglion_cells = initialize_midget_cells(self.cone_bipolar_cells)
        self.parasol_ganglion_cells = initialize_parasol_cells(self.cone_bipolar_cells)
        self.small_bistrat_ganglion_cells = initialize_small_bistratified_cells(self.cone_bipolar_cells)


class Cochlea:
    def __init__(self):
        pass
