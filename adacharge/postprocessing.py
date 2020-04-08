from typing import List
import numpy as np
from copy import deepcopy
from itertools import cycle
from adacharge.datatypes import InfrastructureInfo, SessionInfo
from adacharge.utils import infrastructure_constraints_feasible


def floor_to_set(x: float, allowable_set: np.ndarray, eps=0.05):
    """ Round x down into the allowable set.

    Args:
        x (float): Value to round.
        allowable_set (np.ndarray): Array of the allowable values.
        eps (float): If value is within eps of the next largest allowable value, round up, otherwise round down.

    Returns:
        float: Rounded value.
    """
    return max(allowable_set[allowable_set <= x + eps])


def ceil_to_set(x: float, allowable_set: np.ndarray, eps=0.05):
    """ Round x down into the allowable set.

    Args:
        x (float): Value to round.
        allowable_set (np.ndarray): Array of the allowable values.
        eps (float): If value is within eps of the next lowest allowable value, round down, otherwise round up.

    Returns:
        float: Rounded value.
    """
    return min(allowable_set[allowable_set >= x - eps])


def project_into_continuous_feasible_pilots(rates: np.ndarray, infrastructure: InfrastructureInfo):
    """ Round all values in rates such that they are less than the max_pilot of the corresponding EVSE and greater than
            or equal to 0.

    Args:
        rates (np.ndarray): Schedule of charging rates.
        infrastructure (InfrastructureInfo): Description of the charging infrastructure.

    Returns:
        np.ndarray: Rounded schedule of charging rates.
    """
    new_rates = deepcopy(rates)
    for i, station_id in enumerate(infrastructure.evse_index):
        new_rates[i] = np.minimum(rates[i], infrastructure.max_pilot[i])
    new_rates = np.maximum(new_rates, 0)
    return new_rates


def project_into_discrete_feasible_pilots(rates: np.ndarray, infrastructure: InfrastructureInfo):
    """ Project all values in rates such that they are take one of the allowable pilots for the corresponding EVSE.

    Args:
        rates (np.ndarray): Schedule of charging rates.
        infrastructure (InfrastructureInfo): Description of the charging infrastructure.

    Returns:
        np.ndarray: Rounded schedule of charging rates.
    """
    new_rates = deepcopy(rates)
    N, T = new_rates.shape
    for i, station_id in enumerate(infrastructure.evse_index):
        allowable = np.array(infrastructure.allowable_pilots[i])
        for t in range(T):
            new_rates[i, t] = floor_to_set(rates[i, t], allowable, eps=0.05)
    new_rates = np.maximum(new_rates, 0)
    return new_rates


def index_based_reallocation(rates: np.ndarray, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo,
                             peak_limit: float, index_fn):
    """ Reallocate capacity for first control period up to peak_limit by incrementing the pilot signal to each EV after
            sorting by index_fn.

    Args:
        rates (np.ndarray): Schedule of charging rates.
        active_sessions (List[SessionInfo]): List of active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging infrastructure.
        peak_limit (float):
        index_fn: Function which takes in a SessionInfo object and returns the value of the metric.

    Returns:
        np.ndarray: Schedule of charging rates with reallocation up to peak_limit during the first control period.
    """
    N = len(infrastructure.evse_index)
    sorted_indexes = index_fn(active_sessions, key=index_fn)
    active = rates < infrastructure.max_pilot - 1e-3
    allowable = [infrastructure.allowable_pilots[evse_id] for evse_id in infrastructure.evse_index]

    energy_demands = np.zeros(N)
    for session in active_sessions:
        # Do not record energy demands for sessions not active in the first time interval.
        if session.arrival_offset == 0:
            energy_demands[infrastructure.evse_index.index(session.station_id)] = session.remaining_energy

    for i in cycle(sorted_indexes):
        if not np.any(active):
            break
        if active[i]:
            new_rates = deepcopy(rates)
            new_rates[i] = ceil_to_set(rates[i], allowable[i], 0)
            if np.sum(new_rates) > peak_limit and new_rates < energy_demands and \
                    infrastructure_constraints_feasible(new_rates, infrastructure):
                rates = new_rates
            else:
                active[i] = False