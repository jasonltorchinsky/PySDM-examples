import sys
sys.path.insert(0, '../..') # Get modules from PySDM-examples

import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from datetime import datetime
from pystrict import strict
from PySDM.physics import si
from PySDM.initialisation import spectra

from tools import print_msg

import PySDM_examples.Abdul_Razzak_Ghan_2000
sys.modules['ARG_2000'] = sys.modules['PySDM_examples.Abdul_Razzak_Ghan_2000']
from ARG_2000.run_ARG_parcel import run_parcel
from ARG_2000.data_from_ARG2000_paper import (
    Fig1_N2_obs, Fig1_AF_obs, Fig1_N2_param, Fig1_AF_param,
    #Fig2a_N2_obs, Fig2a_AF_obs, Fig2a_N2_param, Fig2a_AF_param,
    #Fig2b_N2_obs, Fig2b_AF_obs, Fig2b_N2_param, Fig2b_AF_param,
    #Fig3a_sol2_obs, Fig3a_AF_obs, Fig3a_sol2_param, Fig3a_AF_param,
    #Fig3b_sol2_obs, Fig3b_AF_obs, Fig3b_sol2_param, Fig3b_AF_param,
    #Fig4a_rad2_obs, Fig4a_AF_obs, Fig4a_rad2_param, Fig4a_AF_param,
    #Fig4b_rad2_obs, Fig4b_AF_obs, Fig4b_rad2_param, Fig4b_AF_param,
    #Fig5a_w_obs, Fig5a_AF_obs, Fig5a_w_param, Fig5a_AF_param,
    #Fig5b_w_obs, Fig5b_AF_obs, Fig5b_w_param, Fig5b_AF_param
)
from ARG_2000.data_from_CloudMicrophysics_ARG import (
    Fig1_N2_param_jl, Fig1_AF_param_jl_B,
    #Fig2_N2_param_jl, Fig2a_AF_param_jl_B, Fig2b_AF_param_jl_B,
    #Fig3_sol2_param_jl, Fig3a_AF_param_jl_B, Fig3b_AF_param_jl_B,
    #Fig4_rad2_param_jl, Fig4a_AF_param_jl_B, Fig4b_AF_param_jl_B,
    #Fig5_w_param_jl, Fig5a_AF_param_jl_B, Fig5b_AF_param_jl_B,
)

def main():

    print_msg('Starting execution...')
    perf_0_main = perf_counter()
    
    # EXPERIMENT: An air parcel containing two aerosol modes of
    # 100% ammonium sulfate, rising at 0.5 [m s^(-1)]
    n_sd_per_mode = 10 # Number of super-droplets per mode.

    n_trials = 5
    N_1s = np.linspace(100, 5000, n_trials) / si.cm**3 # Number concentration of mode 1.
    AF_S = np.zeros((2, n_trials)) # Activated fraction based on crit. supersat.
    AF_V = np.zeros((2, n_trials)) # Activated fraction based on crit. vol.
    AFerror = np.zeros(n_trials)   # Activated fraction error (?)
    
    w = 0.5 * si.m / si.s # Updraft velocity of the air parcel.

    ammonium_sulfate = Compound(name       = '(NH4)2SO4',
                                molar_mass = 132.14 * si.g / si.mole,
                                density    = 1.77 * si.g / si.cm**3,
                                is_soluble = True,
                                ionic_dissociation_phi = 3)
    sodium_chloride  = Compound(name       = 'NaCl',
                                molar_mass = 58.44 * si.g / si.mole,
                                density    = 2.16 * si.g / si.cm**3,
                                is_soluble = True,
                                ionic_dissociation_phi = 2)
    insoluble        = Compound(name       = 'insoluble',
                                molar_mass = 44 * si.g / si.mole,
                                density    = 1.77 * si.g / si.cm**3,
                                is_soluble = False,
                                ionic_dissociation_phi = 0)
    
    compounds = (ammonium_sulfate, sodium_chloride, insoluble)


    # Create Mode 0 - is same for each loop iteration
    N_0     = 100.0 / si.cm**3 # Number density [cm^(-3)]
    r_0     = 50 * si.nm       # Number mode radius [nm]
    sigma_0 = 2.0              # Geometric std. dev. [N/A]
    mass_fractions_0 = (1.0, 0.0, 0.0) # Mass fractions [N/A]
                                       # * Same order as compounds
    spectrum_0 = spectra.Lognormal(norm_factor = N_0,
                                   m_mode = r_0,
                                   s_geom = sigma_0)
    
    mode_0 = Mode(compounds,
                  mass_fractions = mass_fractions_0,
                  spectrum = spectrum_0)

    # Info for mode 1 common for each loop iteration
    r_1     = 50 * si.nm
    sigma_1 = 2.0
    M_ammonium_sulfate = 1.0  # Mass fraction of (NH4)2SO4
    mass_fractions_1 = (M_ammonium_sulfate, 0.0, (1.0 - M_ammonium_sulfate))
    

    # Run the parcel for various number densities of mode 1
    print_msg('Starting main loop...')
    for ii, N_1 in enumerate(N_1s):
        perf_0 = perf_counter()

        # Finish creating mode 1 with desired number density
        spectrum_1 = spectra.Lognormal(norm_factor = N_1,
                                       m_mode = r_1,
                                       s_geom = sigma_1)
    
        mode_1 = Mode(compounds,
                      mass_fractions = mass_fractions_1,
                      spectrum = spectrum_1)

        modes = (mode_0, mode_1)
        
        output     = run_parcel(w, modes, n_sd_per_mode)
        AF_S[:,ii]  = output.activated_fraction_S
        AF_V[:,ii]  = output.activated_fraction_V
        AFerror[ii] = output.error[0]

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        perf_str = ('Run {:0>2d} of {:0>2d}: ' +
                    'Time Elapsed - {:06.4f} [s]').format(ii, n_trials, perf_diff)
        print_msg(perf_str)

    # Recreate Figure 1 of Razzak00
    fig, ax = plt.subplots(1, 1, sharex = True, figsize = (8, 6))
    # Original Razzak00 starts mode indexing at 1, so our "Mode 1" is
    # their "Mode 2", hence "N2"
    ax.plot(Fig1_N2_obs, Fig1_AF_obs, "ko", label = "ARG 2000 data")
    ax.plot(Fig1_N2_param, Fig1_AF_param, "k-", label = "ARG 2000 param")
        
    ax.plot(Fig1_N2_param_jl, Fig1_AF_param_jl_B, "k--",
            label = "CloudMicrophysics.jl param (B)")
        
    ax.errorbar(N_1s * si.cm**3, AF_S[0,:], yerr = AFerror,
                fmt = 'o', capsize = 4, label = "PySDM, Scrit def")
    ax.errorbar(N_1s * si.cm**3, AF_V[0,:], yerr = AFerror,
                fmt='x', capsize = 2, label = "PySDM, Vcrit def")
    ax.set_ylabel('Mode 0 Activated Fraction')
    ax.set_ylim([0, 1.1])
    
    plt.xlabel('Mode 1 Aerosol Number [cm$^{-3}$]')
    plt.legend(loc = "best")
    plt.savefig('fig_1.pdf')

    perf_f_main = perf_counter()
    perf_diff = perf_f_main - perf_0_main
    perf_str = ('Total execution time - {:06.4f} [s]').format(perf_diff)
    print_msg(perf_str)


@strict
class Compound():
    """
    Hold information about a compound present in an aerosol.
    """

    def __init__(self,
                 name: str,
                 molar_mass: float,
                 density: float,
                 is_soluble: bool,
                 ionic_dissociation_phi: int):

        self.name                   = name
        self.molar_mass             = molar_mass
        self.density                = density
        self.is_soluble             = is_soluble
        self.ionic_dissociation_phi = ionic_dissociation_phi

class Mode():
    """
    Hold information about an aerosol mode.
    """

    def __init__(self,
                 compounds,
                 mass_fractions,
                 spectrum):

        self.compounds = {}
        self.mass_fractions = {}

        ncompounds = len(compounds)
        
        for cc in range(0, ncompounds):
            compound = compounds[cc]
            self.compounds[compound.name]      = compound
            self.mass_fractions[compound.name] = mass_fractions[cc]
            
        self.spectrum = spectrum
    
if __name__ == '__main__':
    main()
