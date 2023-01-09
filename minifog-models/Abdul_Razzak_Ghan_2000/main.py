import sys
sys.path.insert(0, '../..') # Get modules from PySDM-examples

import numpy as np
from matplotlib import pyplot as plt
from atmos_cloud_sim_uj_utils import show_plot
from PySDM.physics import si

import PySDM_examples.Abdul_Razzak_Ghan_2000
sys.modules['ARG_2000'] = sys.modules['PySDM_examples.Abdul_Razzak_Ghan_2000']
from ARG_2000.run_ARG_parcel import run_parcel
from ARG_2000.data_from_ARG2000_paper import (
    Fig1_N2_obs, Fig1_AF_obs, Fig1_N2_param, Fig1_AF_param,
    Fig2a_N2_obs, Fig2a_AF_obs, Fig2a_N2_param, Fig2a_AF_param,
    Fig2b_N2_obs, Fig2b_AF_obs, Fig2b_N2_param, Fig2b_AF_param,
    Fig3a_sol2_obs, Fig3a_AF_obs, Fig3a_sol2_param, Fig3a_AF_param,
    Fig3b_sol2_obs, Fig3b_AF_obs, Fig3b_sol2_param, Fig3b_AF_param,
    Fig4a_rad2_obs, Fig4a_AF_obs, Fig4a_rad2_param, Fig4a_AF_param,
    Fig4b_rad2_obs, Fig4b_AF_obs, Fig4b_rad2_param, Fig4b_AF_param,
    Fig5a_w_obs, Fig5a_AF_obs, Fig5a_w_param, Fig5a_AF_param,
    Fig5b_w_obs, Fig5b_AF_obs, Fig5b_w_param, Fig5b_AF_param
)
from ARG_2000.data_from_CloudMicrophysics_ARG import (
    Fig1_N2_param_jl, Fig1_AF_param_jl_B,
    Fig2_N2_param_jl, Fig2a_AF_param_jl_B, Fig2b_AF_param_jl_B,
    Fig3_sol2_param_jl, Fig3a_AF_param_jl_B, Fig3b_AF_param_jl_B,
    Fig4_rad2_param_jl, Fig4a_AF_param_jl_B, Fig4b_AF_param_jl_B,
    Fig5_w_param_jl, Fig5a_AF_param_jl_B, Fig5b_AF_param_jl_B,
)

def main():
    n_sd_per_mode = 10
    
    N2 = np.linspace(100, 5000, 5) / si.cm**3
    AF_S = np.zeros((2, len(N2)))
    AF_V = np.zeros((2, len(N2)))
    AFerror = np.zeros(len(N2))
    
    w = 0.5 * si.m / si.s
    sol2 = 1.0 # 100% ammonium sulfate
    rad2 = 50.0 * si.nm
    
    for i, N2i in enumerate(N2):
        output = run_parcel(w, sol2, N2i, rad2, n_sd_per_mode)
        AF_S[:,i] = output.activated_fraction_S
        AF_V[:,i] = output.activated_fraction_V
        AFerror[i] = output.error[0]
        
        fig, ax = plt.subplots(1, 1, sharex = True, figsize = (8, 6))
        ax.plot(Fig1_N2_obs, Fig1_AF_obs, "ko", label="ARG 2000 data")
        ax.plot(Fig1_N2_param, Fig1_AF_param, "k-", label="ARG 2000 param")
        
        ax.plot(Fig1_N2_param_jl, Fig1_AF_param_jl_B, "k--", label="CloudMicrophysics.jl param (B)")
        
        ax.errorbar(N2 * si.cm**3, AF_S[0,:], yerr=AFerror, fmt='o', capsize=4, label="PySDM, Scrit def")
        ax.errorbar(N2 * si.cm**3, AF_V[0,:], yerr=AFerror, fmt='x', capsize=2, label="PySDM, Vcrit def")
        ax.set_ylabel('Mode 1 Activated Fraction')
        ax.set_ylim([0,1.1])
        
        plt.xlabel('Mode 2 Aerosol Number (cm$^{-3}$)')
        plt.legend(loc="best")
        plt.savefig('fig_1.pdf')


if __name__ == '__main__':
    main()
