import sys
sys.path.insert(0, '../..') # Get modules from PySDM-examples

import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import os
from PySDM.physics import si
from PySDM.initialisation import spectra

from tools import print_msg

import PySDM_examples.Abdul_Razzak_Ghan_2000
sys.modules['ARG_2000'] = sys.modules['PySDM_examples.Abdul_Razzak_Ghan_2000']
from ARG_2000.Compound import Compound
from ARG_2000.Mode import Mode
from ARG_2000.run_ARG_parcel import run_parcel

def main():

    out_dir = 'figs'
    os.makedirs(out_dir, exist_ok = True)

    print_msg('Starting execution...')
    perf_0_main = perf_counter()
    
    # EXPERIMENT: An air parcel containing two aerosol modes of
    # 100% ammonium sulfate, rising at 0.5 [m s^(-1)]
    n_sd_per_mode = 20 # Number of super-droplets per mode.

    n_trials = 7
    ws   = 2**np.linspace(-5.0, 5.0, n_trials) * si.m / si.s
           # Updraft velocity of the air parcel.
    AF_S = np.zeros((2, n_trials)) # Activated fraction based on crit. supersat.
    AF_V = np.zeros((2, n_trials)) # Activated fraction based on crit. vol.
    AFerror = np.zeros(n_trials)   # Activated fraction error (?)

    RH0 = 0.35
    
    dt = 1 * si.s
    n_steps = 350
    ql    = np.zeros([n_steps, n_trials])
    S_max = np.zeros([n_steps, n_trials])
    RH    = np.zeros([n_steps, n_trials])
    z     = np.zeros([n_steps, n_trials])
    
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
    N_0     = 0.5 * 100.0 / si.cm**3 # Number density [cm^(-3)]
    r_0     = 50 * si.nm       # Number mode radius [nm]
    sigma_0 = 2.0              # Geometric std. dev. [N/A]
    mass_fractions_0 = (0.0, 1.0, 0.0) # Mass fractions [N/A]
                                       # * Same order as compounds
    spectrum_0 = spectra.Lognormal(norm_factor = N_0,
                                   m_mode = r_0,
                                   s_geom = sigma_0)
    
    mode_0 = Mode(compounds,
                  mass_fractions = mass_fractions_0,
                  spectrum = spectrum_0)

    # Code requires multiple modes, so we repeat the same mode twice here
    # and I think it acts like a single mode?
    modes = [mode_0, mode_0]
    

    # Run the parcel for various number densities of mode 1
    print_msg('Starting main loop...')
    for ii, w in enumerate(ws):
        perf_0 = perf_counter()

        output      = run_parcel(w, modes, n_sd_per_mode,
                                 RH0 = RH0, n_steps = n_steps,
                                 dt = dt)
        ql[:,ii]    = output.profile['ql']
        S_max[:,ii] = output.profile['S max']
        RH[:,ii]    = output.profile['RH']
        z[:,ii]     = output.profile['z']
        
        AF_S[:,ii]  = output.activated_fraction_S
        AF_V[:,ii]  = output.activated_fraction_V
        AFerror[ii] = output.error[0]

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        perf_str = ('Run {:0>2d} of {:0>2d}: ' +
                    'Time Elapsed - {:06.4f} [s]').format(ii, n_trials-1, perf_diff)
        print_msg(perf_str)

    # Plot activated fraction of each mode versus parcel vertical velocity
    fig, ax = plt.subplots(1, 1, sharex = True, figsize = (8, 6))
        
    ax.errorbar(ws, AF_S[0,:], yerr = AFerror,
                fmt = 'o', capsize = 4, label = "Mode 0 (NaCl)")
    #ax.errorbar(ws, AF_S[1,:], yerr = AFerror,
    #            fmt = 'x', capsize = 4, label = "Mode 1 (NaCl)")

    ax.set_xlabel('Air Parcel Vertical Velocity [$m\,s^{-1}$]')
    ax.set_xscale('log', base = 2)
    
    ax.set_ylabel('Activated Fraction')
    ax.set_ylim([0, 1.1])
    
    title_str = (('NaCl Number Density {:06.2f}').format(2 * N_0 * si.cm**3) +
                 ' [$cm^{-3}$]\n' +
                 ('Time Elapsed: {:06.2f} [s]').format(dt * n_steps))
    ax.set_title(title_str)
    
    #plt.legend(loc = "best")
    file_name = 'AF_w.png'
    plt.savefig(os.path.join(out_dir, file_name), dpi = 300)

    # Plot liquid water content evolution for each parcel vertical velocity
    fig.clear()
    fig, ax = plt.subplots(1, 1, sharex = True, figsize = (8, 6))

    t = np.arange(0, dt * n_steps, dt)
    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7']
    
    for ii, w in enumerate(ws):
        lbl = ('{:06.4f}').format(w) + ' [$m\,s^{-1}$]' # {} with LaTeX doesn't
                        # p[lay nice with .format()
        ax.plot(t, ql[:,ii], label = lbl, color = colors[ii])
                # g kg^(-1) => g g^(-1)

    ax.set_xlabel('Time [s]')
    
    ax.set_ylabel('Liquid Water Content [$g\,kg^{-1}$]')
    ql_min = np.amax([np.around(np.nanmin(ql), decimals = 2) - 0.1, 0.0])
    ql_max = np.around(np.nanmax(ql), decimals = 2) + 0.1
    ax.set_ylim([ql_min, ql_max])

    title_str = (('NaCl Number Density {:06.2f}').format(2 * N_0 * si.cm**3) +
                 ' [$cm^{-3}$]')
    ax.set_title(title_str)
    
    plt.legend(loc = "best")
    file_name = 'ql_t.png'
    plt.savefig(os.path.join(out_dir, file_name), dpi = 300)

    # Plot relative humidity evolution for each parcel vertical velocity
    fig.clear()
    fig, ax = plt.subplots(1, 1, sharex = True, figsize = (8, 6))

    t = np.arange(0, dt * n_steps, dt)
    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7']
    
    for ii, w in enumerate(ws):
        lbl = ('{:06.4f}').format(w) + ' [$m\,s^{-1}$]' # {} with LaTeX doesn't
                        # p[lay nice with .format()
        ax.plot(t, RH[:,ii], label = lbl, color = colors[ii])

    ax.set_xlabel('Time [s]')
    
    ax.set_ylabel('Relative Humidity')
    RH_min = np.amax([np.around(np.nanmin(RH), decimals = 2) - 0.1, 0.0])
    RH_max = np.around(np.nanmax(RH), decimals = 2) + 0.1
    ax.set_ylim([RH_min, RH_max])

    title_str = (('NaCl Number Density {:06.2f}').format(2 * N_0 * si.cm**3) +
                 ' [$cm^{-3}$]')
    ax.set_title(title_str)
    
    plt.legend(loc = "best")
    file_name = 'RH_t.png'
    plt.savefig(os.path.join(out_dir, file_name), dpi = 300)

    perf_f_main = perf_counter()
    perf_diff = perf_f_main - perf_0_main
    perf_str = ('Total execution time - {:06.4f} [s]').format(perf_diff)
    print_msg(perf_str)
    
if __name__ == '__main__':
    main()
