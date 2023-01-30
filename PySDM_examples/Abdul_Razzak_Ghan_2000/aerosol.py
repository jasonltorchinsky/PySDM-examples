from chempy import Substance
from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si
from pystrict import strict


@strict
class AerosolARG(DryAerosolMixture):
    def __init__(self, modes: tuple):

        # Required that each mode has the same compounds available
        # (Can have mass fraction = 0.0 for compounds to exclude them)
        names                   = list(modes[0].compounds.keys())
        compounds               = list(modes[0].compounds.values())
        molar_masses            = {}
        densities               = {}
        is_solubles             = {}
        ionic_dissociation_phis = {}
        for compound in compounds:
            molar_masses[compound.name] = compound.molar_mass
            densities[compound.name]    = compound.density
            is_solubles[compound.name]  = compound.is_soluble
            ionic_dissociation_phis[compound.name] = compound.ionic_dissociation_phi
            
        super().__init__( # Initialize superclass DryAerosolMixture
            compounds              = names,
            molar_masses           = molar_masses,
            densities              = densities,
            is_soluble             = is_solubles,
            ionic_dissociation_phi = ionic_dissociation_phis,
        )

        self.modes = list({"kappa": self.kappa(mode.mass_fractions),
                           "spectrum": mode.spectrum} for mode in modes)

    def __str__(self):

        # Print compounds present in aerosol, their mass fractions, etc.
        compounds_str              = str(self.compounds)
        molar_masses_str           = str(list(self.molar_masses.values()))
        densities_str              = str(list(self.densities.values()))
        is_soluble_str             = str(list(self.is_soluble.values()))
        ionic_dissociation_phi_str = str(list(self.ionic_dissociation_phi.values()))

        label = ('Compounds:\n  {}\n' +
                 'Molar Masses:\n {}\n' +
                 'Densities:\n {}\n' +
                 'is_soluble:\n {}\n' +
                 'ionic_dissociation_phi:\n {}').format(compounds_str,
                                                        molar_masses_str,
                                                        densities_str,
                                                        is_soluble_str,
                                                        ionic_dissociation_phi_str)
        
        return label
        


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
