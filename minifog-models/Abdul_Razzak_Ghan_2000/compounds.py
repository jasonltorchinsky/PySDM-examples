import sys
sys.path.insert(0, '../..') # Get modules from PySDM-examples

from PySDM.physics import si

from tools import print_msg

import PySDM_examples.Abdul_Razzak_Ghan_2000
sys.modules['ARG_2000'] = sys.modules['PySDM_examples.Abdul_Razzak_Ghan_2000']
from ARG_2000.Compound import Compound


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
