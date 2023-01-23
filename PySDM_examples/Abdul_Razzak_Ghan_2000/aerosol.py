from chempy import Substance
from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si
from pystrict import strict


@strict
class AerosolARG(DryAerosolMixture):
    def __init__(self,
                 M2_sol: float = 0, # Fraction of externally mixed mode 2
                 M2_N:   float = 100 / si.cm**3,
                 M2_rad: float = 50 * si.nm,):
        super().__init__( # Initialize superclass DryAerosolMixture
            compounds = ("(NH4)2SO4", "NaCl", "insoluble"),
            molar_masses = {
                "(NH4)2SO4": 132.14 * si.g / si.mole,
                "NaCl"     : 58.44 * si.g / si.mole,
                "insoluble": 44 * si.g / si.mole,
            },
            densities = {
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "NaCl"     : 2.16 * si.g / si.cm**3,
                "insoluble": 1.77 * si.g / si.cm**3,
            },
            is_soluble = {"(NH4)2SO4": True,
                          "NaCl":      True,
                          "insoluble": False},
            ionic_dissociation_phi = {"(NH4)2SO4": 3,
                                      "NaCl":      2,
                                      "insoluble": 0},
        )
        self.modes = (
            { # Entirely (NH4)2SO4
                "kappa": self.kappa({"(NH4)2SO4": 1.0,
                                     "NaCl":      0.0,
                                     "insoluble": 0.0}),
                "spectrum": spectra.Lognormal(
                    norm_factor = 100.0 / si.cm**3,
                    m_mode = 50.0 * si.nm,
                    s_geom = 2.0
                ),
            },
            { # Externally mixed (NH4)2SO4, Arbitrary insoluble compound
                "kappa": self.kappa({"(NH4)2SO4": M2_sol,
                                     "NaCl":      0.0,
                                     "insoluble": (1 - M2_sol)}),
                "spectrum": spectra.Lognormal(
                    norm_factor = M2_N,
                    m_mode = M2_rad,
                    s_geom = 2.0
                ),
            },
        )


@strict
class AerosolWhitby(DryAerosolMixture):
    def __init__(self):
        nuclei = {"(NH4)2SO4": 1.0}
        accum = {"(NH4)2SO4": 1.0}
        coarse = {"(NH4)2SO4": 1.0}

        super().__init__(
            ionic_dissociation_phi={"(NH4)2SO4": 3},
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole
            },
            densities={"(NH4)2SO4": 1.77 * si.g / si.cm**3},
            compounds=("(NH4)2SO4",),
            is_soluble={"(NH4)2SO4": True},
        )
        self.modes = (
            {
                "kappa": self.kappa(nuclei),
                "spectrum": spectra.Lognormal(
                    norm_factor=1000.0 / si.cm**3, m_mode=0.008 * si.um, s_geom=1.6
                ),
            },
            {
                "kappa": self.kappa(accum),
                "spectrum": spectra.Lognormal(
                    norm_factor=800 / si.cm**3, m_mode=0.034 * si.um, s_geom=2.1
                ),
            },
            {
                "kappa": self.kappa(coarse),
                "spectrum": spectra.Lognormal(
                    norm_factor=0.72 / si.cm**3, m_mode=0.46 * si.um, s_geom=2.2
                ),
            },
        )
