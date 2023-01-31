class Mode():
    """
    Hold information about an aerosol mode.
    """

    def __init__(self,
                 compounds,      # List of Compounds in mode (see Compounds class)
                 mass_fractions, # Mass fractions of each compound in mode
                 spectrum):      # Particle size spectrum of compounds in mode
                                 # (see spectra from PySDM.initialisation)

        self.compounds = {}
        self.mass_fractions = {}

        ncompounds = len(compounds)
        
        for cc in range(0, ncompounds):
            compound = compounds[cc]
            self.compounds[compound.name]      = compound
            self.mass_fractions[compound.name] = mass_fractions[cc]
            
        self.spectrum = spectrum
