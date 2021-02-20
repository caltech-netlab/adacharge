import bisect
from typing import List
import numpy as np
from copy import deepcopy
from itertools import cycle
from acnportal.acnsim.interface import *
from .utils import infrastructure_constraints_feasible


def floor_to_set(x: float, allowable_set: np.ndarray, eps=0.05):
    """Round x down into the allowable set. If x is less than minimum value
        in allowable_set, clip to the minimum.

    Args:
        x (float): Value to round.
        allowable_set (np.ndarray): Array of the allowable values.
        eps (float): If value is within eps of the next largest allowable
            value, round up, otherwise round down.

    Returns:
        float: Rounded value.
    """
    pos = bisect.bisect_left(allowable_set, x + eps)
    if pos < len(allowable_set):
        if x == allowable_set[pos]:
            return x
    if pos == 0:
        return allowable_set[0]
    if pos == len(allowable_set):
        return allowable_set[-1]
    return allowable_set[pos - 1]


def ceil_to_set(x: float, allowable_set: np.ndarray, eps=0.05):
    """Round x up into the allowable set. If x is greater than maximum value
        in allowable_set, clip to the maximum.

    Args:
        x (float): Value to round.
        allowable_set (np.ndarray): Array of the allowable values.
        eps (float): If value is within eps of the next lowest allowable value,
            round down, otherwise round up.

    Returns:
        float: Rounded value.
    """
    pos = bisect.bisect_right(allowable_set, x - eps)
    if pos > 0:
        if x == allowable_set[pos - 1]:
            return x
    if pos == 0:
        return allowable_set[0]
    if pos == len(allowable_set):
        return allowable_set[-1]
    return allowable_set[pos]


def increment_in_set(x: float, allowable_set: np.ndarray):
    """Increment x to the next largest value in allowable set. If x is
        greater than maximum value in allowable_set, clip to the maximum.

    Args:
        x (float): Value to round.
        allowable_set (np.ndarray): Array of the allowable values.

    Returns:
        float: Rounded value.
    """
    pos = bisect.bisect_right(allowable_set, x)
    if pos == 0:
        return allowable_set[0]
    if pos == len(allowable_set):
        return allowable_set[-1]
    return allowable_set[pos]


def project_into_continuous_feasible_pilots(
    rates: np.ndarray, infrastructure: InfrastructureInfo
):
    """Round all values in rates such that they are less than the max_pilot of the corresponding EVSE and greater than
            or equal to 0.

    Args:
        rates (np.ndarray): Schedule of charging rates.
        infrastructure (InfrastructureInfo): Description of the charging infrastructure.

    Returns:
        np.ndarray: Rounded schedule of charging rates.
    """
    new_rates = deepcopy(rates)
    for i in range(infrastructure.num_stations):
        new_rates[i] = np.minimum(rates[i], infrastructure.max_pilot[i])
    new_rates = np.maximum(new_rates, 0)
    return new_rates


def project_into_discrete_feasible_pilots(
    rates: np.ndarray, infrastructure: InfrastructureInfo
):
    """Project all values in rates such that they are take one of the
        allowable pilots for the corresponding EVSE.

    Args:
        rates (np.ndarray): Schedule of charging rates.
        infrastructure (InfrastructureInfo): Description of the charging
            infrastructure.

    Returns:
        np.ndarray: Rounded schedule of charging rates.
    """
    new_rates = deepcopy(rates)
    N, T = new_rates.shape
    for i in range(infrastructure.num_stations):
        allowable = np.array(infrastructure.allowable_pilots[i])
        for t in range(T):
            new_rates[i, t] = floor_to_set(rates[i, t], allowable, eps=0.05)
    new_rates = np.maximum(new_rates, 0)
    return new_rates


def index_based_reallocation(
    rates: np.ndarray,
    active_sessions: List[SessionInfo],
    infrastructure: InfrastructureInfo,
    peak_limit: float,
    sort_fn,
    interface: Interface,
):
    """Reallocate capacity for first control period up to peak_limit by
        incrementing the pilot signal to each EV after sorting by index_fn.

    Args:
        rates (np.ndarray): Schedule of charging rates.
        active_sessions (List[SessionInfo]): List of active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging
            infrastructure.
        peak_limit (float):
        sort_fn: Function which takes in a SessionInfo object and returns the
            value of the metric.
        interface (Interface): Interface to information about the environment.


    Returns:
        np.ndarray: Schedule of charging rates with reallocation up to peak_limit during the first control period.
    """
    sorted_sessions = sort_fn(active_sessions, interface)
    sorted_indexes = [
        infrastructure.get_station_index(s.station_id) for s in sorted_sessions
    ]
    active = np.zeros(infrastructure.num_stations, dtype=bool)
    ub = np.zeros(infrastructure.num_stations)
    for session in active_sessions:
        # Do not record energy demands for sessions not active in the first
        # time interval, as these could be future sessions at the same station.
        if session.arrival_offset == 0:
            i = infrastructure.station_ids.index(session.station_id)
            active[i] = True
            ub[i] = min(
                [
                    interface.remaining_amp_periods(session),
                    session.max_rates[0],
                    infrastructure.max_pilot[i],
                ]
            )

    for i in cycle(sorted_indexes):
        if not np.any(active):
            break

        if active[i]:
            if rates[i, 0] >= ub[i]:
                active[i] = False
                continue
            new_rates = deepcopy(rates[:, 0])
            new_rates[i] = increment_in_set(
                rates[i, 0], infrastructure.allowable_pilots[i]
            )
            if (
                np.sum(new_rates) <= peak_limit
                and new_rates[i] <= ub[i]
                and infrastructure_constraints_feasible(new_rates, infrastructure)
            ):
                rates[:, 0] = new_rates
            else:
                active[i] = False
    return rates


def diff_based_reallocation(
    rates: np.ndarray,
    active_sessions: List[SessionInfo],
    infrastructure: InfrastructureInfo,
    interface: Interface,
):
    """Reallocate capacity for first control period by incrementing the
    pilot signal to each EV. Ordering is determined by the difference between
    the originally allocated current and the quantized current.

    Args:
        rates (np.ndarray): Schedule of charging rates.
        active_sessions (List[SessionInfo]): List of active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging
            infrastructure.
        interface (Interface): Interface to information about the environment.


    Returns:
        np.ndarray: Schedule of charging rates with reallocation.
    """
    init_rates = rates[:, 0]
    peak_limit = init_rates.sum()
    rounded_rates = project_into_discrete_feasible_pilots(rates, infrastructure)

    def metric(session):
        i = infrastructure.get_station_index(session.station_id)
        return -(init_rates[i] - rounded_rates[i, 0])

    sorted_sessions = sorted(active_sessions, key=metric)
    sorted_indexes = [
        infrastructure.get_station_index(s.station_id) for s in sorted_sessions
    ]
    active = np.zeros(infrastructure.num_stations, dtype=bool)
    ub = np.zeros(infrastructure.num_stations)
    for session in active_sessions:
        # Do not record energy demands for sessions not active in the first
        # time interval, as these could be future sessions at the same station.
        if session.arrival_offset == 0:
            i = infrastructure.station_ids.index(session.station_id)
            active[i] = True
            ub[i] = min(
                [
                    interface.remaining_amp_periods(session),
                    session.max_rates[0],
                    infrastructure.max_pilot[i],
                ]
            )

    for i in cycle(sorted_indexes):
        if not np.any(active):
            break

        if active[i]:
            if rounded_rates[i, 0] >= ub[i]:
                active[i] = False
                continue
            new_rates = deepcopy(rounded_rates[:, 0])
            new_rates[i] = increment_in_set(
                rounded_rates[i, 0], infrastructure.allowable_pilots[i]
            )
            if (
                np.sum(new_rates) <= peak_limit
                and new_rates[i] <= ub[i]
                and infrastructure_constraints_feasible(new_rates, infrastructure)
            ):
                rounded_rates[:, 0] = new_rates
            else:
                active[i] = False
    return rounded_rates
