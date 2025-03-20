class Retina:
    """
        Represents a digital retina based on a 2D hexagonal surface.
    """

    def __init__(self):
        self.photoreceptor_cells = None
        self.surface_radius = None
        self.cone_threshold = None

        self.horizontal_cells = None
        self.bipolar_cells = None
        self.amacrine_cells = None
        self.ganglion_cells = None

    def init_photoreceptors(self, cells, surface_radius, cone_threshold):
        self.photoreceptor_cells = cells
        self.surface_radius = surface_radius
        self.cone_threshold = cone_threshold

    def init_horizontal_cells(self, cells):
        self.horizontal_cells = cells

    def init_bipolar_cells(self, cells):
        self.bipolar_cells = cells

    def init_amacrine_cells(self, cells):
        self.amacrine_cells = cells

    def init_ganglion_cells(self, cells):
        self.ganglion_cells = cells
