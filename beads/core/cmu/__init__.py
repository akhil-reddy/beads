from beads.core.cmu.sequencing.combine.audio import *
from beads.core.cmu.sequencing.combine.vision import *
from beads.core.cmu.sequencing.receive.audio import *
from beads.core.cmu.sequencing.receive.vision import *
from beads.core.cmu.sequencing.transforms.audio import *
from beads.core.cmu.sequencing.transforms.vision import *
from beads.core.cmu.transportation.audio import *
from beads.core.cmu.transportation.vision import *


class Retina:
    """
        Represents a digital retina based on a 2D hexagonal surface.
    """

    def __init__(self):
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
    Function initializes the photoreceptors and organizes them.
    """
    def init_photoreceptors(self):
        self.photoreceptor_cells = initialize_photoreceptors()

    """
    Function initializes the horizontal cells and organizes them.
    """
    def init_horizontal_cells(self):
        self.horizontal_cells = initialize_horizontal_cells(self.photoreceptor_cells)

    """
    Function initializes the rod bipolar cells and organizes them.
    """
    def init_rod_bipolar_cells(self):
        self.rod_bipolar_cells = initialize_rod_bipolar_cells(self.photoreceptor_cells)

    """
    Function initializes the AII amacrine cells and organizes them.
    """
    def init_aii_amacrine_cells(self):
        self.aii_amacrine_cells = initialize_aii_amacrine_cells(self.rod_bipolar_cells)

    """
    Function initializes the cone bipolar cells and organizes them.
    """
    def init_cone_bipolar_cells(self):
        self.cone_bipolar_cells = initialize_cone_bipolar_cells(self.horizontal_cells, self.aii_amacrine_cells)

    """
    Function initializes the star amacrine cells and organizes them.
    """
    def init_star_amacrine_cells(self):
        self.star_amacrine_cells = initialize_starburst_amacrine_cells(self.cone_bipolar_cells)

    """
    Function initializes the ganglion cells and organizes them.
    """
    def init_ganglion_cells(self):
        self.dsgc = initialize_DSGCs(self.star_amacrine_cells)
        self.midget_ganglion_cells = initialize_midget_cells(self.cone_bipolar_cells)
        self.parasol_ganglion_cells = initialize_parasol_cells(self.cone_bipolar_cells)
        self.small_bistrat_ganglion_cells = initialize_small_bistratified_cells(self.cone_bipolar_cells)


class Cochlea:
    """
        Represents a digital cochlea based on a sampling rate (fs) and signal.
    """
    def __init__(self, fs):
        self.fs = fs

        self.outer_ear = None

        self.basilar_membrane = None
        self.ohc_cells = None
        self.moc = None

        self.ihc_cells = None
        self.ribbon_synapse = None

        self.anf_spike_trains = None

    """
    Function initializes the outer ear with pinna and ear canal.
    """
    def init_outer_ear(self):
        self.outer_ear = initialize_outer_ear(self.fs)

    """
    Function initializes the OHC cells and organizes them.
    """
    def init_ohc_cells(self, segs):
        self.basilar_membrane = BasilarMembrane()
        self.ohc_cells = [OuterHairCell(seg) for seg in segs]
        self.moc = MedialOlivocochlear(self.ohc_cells)

    """
    Function initializes the IHC cells and organizes them.
    """
    def init_ihc_cells(self, segs):
        self.ihc_cells = [InnerHairCell(seg) for seg in segs]
        self.ribbon_synapse = RibbonSynapse()

    """
    Function initializes the auditory nerve fiber.
    """
    def init_anf(self, vesicle_releases):
        self.anf_spike_trains = generate_anf_spike_trains(vesicle_releases, self.fs)

# TODO: Add classes for cortices

