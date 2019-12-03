import numpy as np
from adacharge import AdaCharge, SOC, AFFINE
import cvxpy as cp
from copy import deepcopy
from .cvx_utils import *
from acnportal import acnsim

J1772_MIN = 6

class AdaChargePostProcessor:
    def __init__(self, interface, const_type=SOC, solver=None, eta=1e-5, integer_program=False):
        self.interface = interface
        self.const_type = const_type
        self.solver = solver
        self.eta = eta
        self.integer_program = False

    def obj(self, rates, target, eta):
        return -cp.sum_squares(cp.sum(rates) - np.sum(target)) - eta*cp.sum_squares(rates - target)

    def _min_allowable_rate(self, ev):
        continuous, allowable_rates = self.interface.allowable_pilot_signals(ev.station_id)
        if continuous:
            return J1772_MIN
        else:
            return max(J1772_MIN, min(x for x in allowable_rates if x > 0))

    def process(self, target, active_evs, verbose=False, solver=None):
        if len(active_evs) == 0:
            return {}
        network_constraints = self.interface.get_constraints()
        active_evses = set(ev.station_id for ev in active_evs)
        evse_indexes = [evse_id for evse_id in network_constraints.evse_index if evse_id in active_evses]

        target_vector = np.array([target[evse_id][0] for evse_id in evse_indexes])
        rates = cp.Variable(target_vector.shape, name='rates', integer=self.integer_program)
        constraints = {}

        max_rates = {ev.session_id: min([self.interface.max_pilot_signal(ev.station_id),
                                         self.interface.remaining_amp_periods(ev)]) for ev in active_evs}
        rates_ub = np.zeros(rates.shape)
        rates_lb = np.zeros(rates.shape)
        for ev in active_evs:
            i = evse_indexes.index(ev.station_id)
            rates_ub[i] = max_rates[ev.session_id]
            rates_lb[i] = self._min_allowable_rate(ev)
        constraints['Rate Upper Bounds'] = rates <= rates_ub
        constraints['Rate Lower Bounds'] = rates >= rates_lb

        phases = {evse_id: self.interface.evse_phase(evse_id) for evse_id in evse_indexes}
        constraints.update(infrastructure_constraints(rates, network_constraints, evse_indexes, self.const_type, phases))

        objective = cp.Maximize(self.obj(rates, target_vector, self.eta))
        prob = cp.Problem(objective, list(constraints.values()))

        _ = prob.solve(verbose=verbose, solver=solver)
        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError('Problem Infeasible.')

        active_evses = set(ev.station_id for ev in active_evs)
        network_constraints = self.interface.get_constraints()
        evse_indexes = [evse_id for evse_id in network_constraints.evse_index if evse_id in active_evses]
        return {evse_id: [int(np.clip(rates[j].value,
                                 a_min=rates_lb[j],
                                 a_max=rates_ub[j]))]
                          for j, evse_id in enumerate(evse_indexes)}

