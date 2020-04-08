from typing import List
from copy import deepcopy
import numpy as np

from acnportal.algorithms import Rampdown
from adacharge.datatypes import SessionInfo, InfrastructureInfo
from adacharge.utils import infrastructure_constraints_feasible


def enforce_evse_pilot_limit(active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo):
    """ Update the max_rates vector for each session to be less than the max pilot supported by its EVSE.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging infrastructure.

    Returns:
        List[SessionInfo]: Active sessions with max_rates updated to be at most the max_pilot of the corresponding EVSE.
    """
    new_sessions = deepcopy(active_sessions)
    for session in new_sessions:
        i = infrastructure.get_station_index(session.station_id)
        session.max_rates = np.minimum(session.max_rates, infrastructure.max_pilot[i])
    return new_sessions


def reconcile_max_and_min(session: SessionInfo):
    """ Modify session.max_rates[t] to equal session.min_rates[t] for times when max_rates[t] < min_rates[t]

    Args:
        session (SessionInfo): Session object.

    Returns:
        SessionInfo: session modified such that max_rates[t] is never less than min_rates[t]
    """
    new_sess = deepcopy(session)
    new_sess.max_rates[new_sess.max_rates < new_sess.min_rates] = new_sess.min_rates[new_sess.max_rates < new_sess.min_rates]
    return new_sess


def expand_max_min_rates(active_sessions: List[SessionInfo]):
    """ Expand max_rates and min_rates to vectors if they are scalars.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.

    Returns:
        List[SessionInfo]: Active sessions with max_rates and min_rates expanded into vectors of length remaining_time.
    """
    new_sessions = deepcopy(active_sessions)
    for session in new_sessions:
        if np.isscalar(session.max_rates):
            session.max_rates = np.array([session.max_rates] * session.remaining_time)
        if np.isscalar(session.min_rates):
            session.min_rates = np.array([session.min_rates] * session.remaining_time)
    return new_sessions


def apply_rampdown(rampdown: Rampdown, active_sessions: List[SessionInfo]):
    """ Update max_rate in each SessionInfo object to account for rampdown.

        If rampdown max_rate is less than min_rate, max_rate is set equal to min_rate.

    Args:
        rampdown (Rampdown): Rampdown-like object which returns new maximum charging rates for each session.
        active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.

    Returns:
        List[SessionInfo]: Active sessions with updated max_rate using rampdown.
    """
    new_sessions = expand_max_min_rates(active_sessions)
    rampdown_limits = rampdown.get_maximum_rates(active_sessions)
    for j, session in enumerate(new_sessions):
        session.max_rates = np.minimum(session.max_rates, rampdown_limits.get(session.station_id, float('inf')))
        new_sessions[j] = reconcile_max_and_min(session)
    return new_sessions


def apply_minimum_charging_rate(active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo,
                                override=float('inf')):
    """ Modify active_sessions so that min_rates[0] is equal to the greater of the session minimum rate and the EVSE
        minimum pilot.
    
    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging infrastructure.
        override (float): Alternative minimum pilot which overrides the EVSE minimum if the EVSE minimum is less than
            override.

    Returns:
        List[SessionInfo]: Active sessions with updated minimum charging rate for the first control period.
    """
    session_queue = sorted(active_sessions, key=lambda x: x.arrival)
    session_queue = expand_max_min_rates(session_queue)
    rates = np.zeros(len(infrastructure.evse_index))
    for j, session in enumerate(session_queue):
        i = infrastructure.evse_index.index(session.station_id)
        rates[i] = min(infrastructure.min_pilot[session.station_id], override)
        if infrastructure_constraints_feasible(rates, infrastructure):
            session.min_rates[0] = max(rates[i], session.min_rates[0])
            session_queue[j] = reconcile_max_and_min(session)
        else:
            rates[i] = 0
            session.min_rates[0] = 0
            session.max_rates[0] = 0
    return session_queue

