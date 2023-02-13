import sys
sys.path.insert(0, '../..') # Get modules from PySDM-examples

import numpy as np
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from time import perf_counter
import os
from PySDM.physics import si
from PySDM.initialisation import spectra

from tools import print_msg
from compounds import ammonium_sulfate, sodium_chloride, insoluble

import PySDM_examples.Abdul_Razzak_Ghan_2000
sys.modules['ARG_2000'] = sys.modules['PySDM_examples.Abdul_Razzak_Ghan_2000']
from ARG_2000.Compound import Compound
from ARG_2000.Mode import Mode
from ARG_2000.run_ARG_parcel import run_parcel

def main():
    print_msg('Starting execution...')
    perf_0_main = perf_counter()
    
    out_dir = 'figs'
    os.makedirs(out_dir, exist_ok = True)
    
    # EXPERIMENT: An air parcel containing two aerosol modes of
    # 100% NaCl rising at various speeds with various number of particles, etc.

    # Run parameters
    dt = 1 * si.s
    n_steps = 350
    RH0 = 0.95              # Initial Relative Humidity [N/A]
    Ns  = np.arange(5000, 30000, 5000) / si.cm**3
                            # Number concentrations of "both" modes [cm^(-3)]
    r   = 50 * si.nm        # Mean radius of "both" modes [nm]
    sigma = 20   # Geometric standard deviation of "both" modes [nm^2]
    ws  = np.arange(0.1, 1.1, 0.2) * si.m / si.s
                            # Vertical velocities of air parcel [m s^(-1)]

    ntrials = np.size(Ns) * np.size(ws)
    
    n_sd_per_mode = 40 # Number of super-droplets per mode.
    compounds = (ammonium_sulfate, sodium_chloride, insoluble)

    # Set up the plots
    t = np.arange(0, dt * n_steps, dt)
    
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    styles = ['solid', (0, (5, 10)), (0, (5, 5)), (0, (5, 1)),
              (0, (1, 10)), (0, (1, 5)), (0, (1, 10)), (0, (3, 5, 1, 5, 1, 5))]
    nfigs = 3
    figs  = [None] * nfigs
    axs   = [None] * nfigs

    ylabels = ['Liquid Water Content [$g\,kg^{-1}$]',
               'Relative Humidity', 'Air Parcel Height [$m$]']
    for nn in range(0, nfigs):
        figs[nn], axs[nn] = plt.subplots(1, 1, sharex = True, figsize = (8, 6))

        axs[nn].set_title('NaCl Aerosol Evolution')
        axs[nn].set_xlabel('Time [s]')
        axs[nn].set_ylabel(ylabels[nn])

        
    # Run the main loop
    trial = 0
    for ii, N in enumerate(Ns):
        # Set plot parameters
        color = colors[ii]
        N_str = '{}'.format(N / (si.cm**(-3))) + ' [$cm^{-3}$]'
        
        # Create Mode 0 - is same for each loop iteration
        N_0     = N/2     # Number density [cm^(-3)] (We repeat this mode,
                                                    # so half the number density)
        r_0     = r       # Number mode radius [nm]
        sigma_0 = sigma   # Geometric std. dev. [N/A]
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
        
        for jj, w in enumerate(ws):
            # Set plot parameters
            style = styles[jj]
            
            perf_0 = perf_counter()

            output = run_parcel(w, modes, n_sd_per_mode,
                                RH0 = RH0,
                                mass_of_dry_air = 1 * si.kg,
                                n_steps = n_steps,
                                dt = dt)

            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            perf_str = ('Run {:0>2d} of {:0>2d}: ' +
                        'Time Elapsed - {:06.4f} [s]').format(trial, ntrials-1,
                                                              perf_diff)
            print_msg(perf_str)
            
            ql    = output.profile['ql']
            RH    = output.profile['RH']
            z     = output.profile['z']

            out_vars = [ql, RH, z]

            for nn in range(0, nfigs):
                axs[nn].plot(t, out_vars[nn],
                             color = color,
                             linestyle = style)

            trial += 1

    # Create legend labels for each plot
    patches = [None] * np.size(Ns)
    for ii, N in enumerate(Ns):
        color = colors[ii]
        N_str = '{}'.format(N / (si.cm**(-3))) + ' $cm^{-3}$'
        patches[ii] = mpatches.Patch(color = color, label = N_str)

    lines = [None] * np.size(ws)
    for jj, w in enumerate(ws):
        style = styles[jj]
        w_str = '{:02.1f}'.format(w / (si.m / si.s)) + ' $m\,s^{-1}$'
        lines[jj] = mlines.Line2D([], [],
                                  color = 'k', linestyle = style,
                                  label = w_str)
            
    # Finish making, saving plots
    file_names = ['ql.png', 'RH.png', 'z.png']
    handles = patches + lines
    for nn in range(0, nfigs):
        axs[nn].legend(handles = handles)

        file_name = os.path.join(out_dir, file_names[nn])
        plt.figure(figs[nn])
        plt.savefig(file_name, dpi = 300)
            

    perf_f_main = perf_counter()
    perf_diff = perf_f_main - perf_0_main
    perf_str = ('Total execution time - {:06.4f} [s]').format(perf_diff)
    print_msg(perf_str)
 
'''
    ax.set_xlabel('Time [s]')
    
    ax.set_ylabel(' ')
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

    
'''

if __name__ == '__main__':
    main()
