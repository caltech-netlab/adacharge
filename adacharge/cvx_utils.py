import numpy as np
import cvxpy as cp

SOC = 'SOC'
AFFINE = 'AFFINE'


def rate_constraints(rates, active_evs, evse_indexes, max_pilots, min_pilots=None):
    constraints = {}
    rates_ub = np.zeros(rates.shape)
    rates_lb = np.zeros(rates.shape)
    for ev in active_evs:
        i = evse_indexes.index(ev.station_id)
        rates_ub[i, ev.arrival:ev.departure] = max_pilots[ev.session_id]
        if min_pilots is not None:
            rates_lb[i, ev.arrival:ev.departure] = min_pilots[ev.session_id]
    constraints['Rate Upper Bounds'] = rates <= rates_ub
    constraints['Rate Lower Bounds'] = rates >= rates_lb
    return constraints


def energy_constraints(rates, active_evs, evse_indexes, remaining_demand, energy_equality=False):
    constraints = {}
    for ev in active_evs:
        # Constraint on the energy delivered to each EV
        i = evse_indexes.index(ev.station_id)
        e = cp.Parameter(nonneg=True, name='{0}_energy_request'.format(ev.session_id))
        e.value = remaining_demand[ev.session_id]
        constraint_id = 'Energy Constraint {0}'.format(ev.session_id)
        if energy_equality:
            constraints[constraint_id] = cp.sum(rates[i, ev.arrival:ev.departure]) == e
        else:
            constraints[constraint_id] = cp.sum(rates[i, ev.arrival:ev.departure]) <= e
    return constraints


def infrastructure_constraints(rates, network_constraints, evse_indexes, const_type, phases=None):
    if network_constraints.constraint_matrix is None:
        return []
    constraints = {}
    trimmed_constraints = network_constraints.constraint_matrix[:,
                          np.isin(network_constraints.evse_index, evse_indexes)]
    inactive_mask = ~np.all(trimmed_constraints == 0, axis=1)
    trimmed_constraints = trimmed_constraints[inactive_mask]
    trimmed_constraint_ids = np.array(network_constraints.constraint_index)[inactive_mask]

    if const_type == SOC:
        if phases is None:
            raise ValueError('phases is required when using SOC infrastructure constraints.')
        phase_vector = np.array([np.deg2rad(phases[evse_id]) for evse_id in evse_indexes])
        for j in range(trimmed_constraints.shape[0]):
            v = np.stack(
                [trimmed_constraints[j, :] * np.cos(phase_vector), trimmed_constraints[j, :] * np.sin(phase_vector)])
            constraints[str(trimmed_constraint_ids[j])] = cp.norm(v * rates, axis=0) <= \
                                                          network_constraints.magnitudes[inactive_mask][j]
    elif const_type == AFFINE:
        for j in range(trimmed_constraints.shape[0]):
            v = np.abs(trimmed_constraints[j, :])
            constraints[str(trimmed_constraint_ids[j])] = v * rates <= network_constraints.magnitudes[inactive_mask][j]
    else:
        raise ValueError(
            'Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.'.format(const_type))
    return constraints
