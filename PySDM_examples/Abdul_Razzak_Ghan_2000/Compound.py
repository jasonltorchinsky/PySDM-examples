from pystrict import strict
@strict
class Compound():
    """
    Hold information about a compound present in an aerosol mode.
    """

    def __init__(self,
                 name: str,
                 molar_mass: float,
                 density: float,
                 is_soluble: bool,
                 ionic_dissociation_phi: int):

        self.name                   = name              # Name of compound
        self.molar_mass             = molar_mass        # Molar mass of compound
        self.density                = density           # Bulk density of compound
        self.is_soluble             = is_soluble        # Is compound soluble?
        self.ionic_dissociation_phi = ionic_dissociation_phi
             # Number of ions compound dissociates into
