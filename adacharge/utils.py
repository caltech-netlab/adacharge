import numpy as np
from acnportal.acnsim.interface import InfrastructureInfo


def infrastructure_constraints_feasible(rates, infrastructure: InfrastructureInfo):
    phase_in_rad = np.deg2rad(infrastructure.phases)
    for j, v in enumerate(infrastructure.constraint_matrix):
        a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
        line_currents = np.linalg.norm(a @ rates, axis=0)
        if not np.all(line_currents <= infrastructure.constraint_limits[j] + 1e-7):
            return False
    return True
