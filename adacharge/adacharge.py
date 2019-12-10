import numpy as np
from acnportal.algorithms import BaseAlgorithm
import cvxpy as cp
from copy import deepcopy

from .cvx_utils import *


class AdaChargeBase(BaseAlgorithm):
    def __init__(self, obj_config, const_type=SOC, energy_equality=False, solver=None, max_recomp=None, offline=False, events=None,
                 post_processor=None):
        super().__init__()
        self.obj_config = obj_config
        if len(self.obj_config) < 1:
            raise ValueError('Please supply a non-empty obj_config.')
        self.offline = offline
        self.evs = []
        self.internal_schedule = None
        if self.offline:
            self.max_recompute = 1
            if events is None:
                raise ValueError('Error. Argument evs is required when solving offline')
            else:
                for event in events._queue:
                    if event[1].type == 'Plugin':
                        self.evs.append(deepcopy(event[1].ev))
        else:
            self.max_recompute = max_recomp
        self.const_type = const_type
        self.energy_equality = energy_equality
        self.solver = solver
        self.post_processor = post_processor

    def obj(self, rates, active_evs):
        return sum(x[0]*x[1](rates, active_evs, self.interface) if len(x) == 2 else
                   x[0]*x[1](rates, active_evs, self.interface, **x[2]) for x in self.obj_config)

    def individual_rate_constraints(self, rates, active_evs, evse_indexes):
        max_rates = {ev.session_id: self.interface.max_pilot_signal(ev.station_id) for ev in active_evs}
        return rate_constraints(rates, active_evs, evse_indexes, max_rates)

    def energy_delivered_constraints(self, rates, active_evs, evse_indexes):
        remaining_demands = {ev.session_id: self.interface.remaining_amp_periods(ev) for ev in active_evs}
        return energy_constraints(rates, active_evs, evse_indexes, remaining_demands, self.energy_equality)

    def infrastructure_constraints(self, rates, evse_indexes):
        network_constraints = self.interface.get_constraints()
        phases = {evse_id: self.interface.evse_phase(evse_id) for evse_id in evse_indexes}
        return infrastructure_constraints(rates, network_constraints, evse_indexes, self.const_type, phases)

    def _build_problem(self, active_evs, offset_time):
        if len(active_evs) == 0:
            return {}

        network_constraints = self.interface.get_constraints()
        active_evses = set(ev.station_id for ev in active_evs)
        evse_indexes = [evse_id for evse_id in network_constraints.evse_index if evse_id in active_evses]
        for ev in active_evs:
            ev.arrival = max(0, ev.arrival - offset_time)
            ev.departure -= offset_time
        max_t = max(ev.departure for ev in active_evs) + 1

        rates = cp.Variable((len(evse_indexes), max_t), name='rates')
        constraints = {}
        constraints.update(self.individual_rate_constraints(rates, active_evs, evse_indexes))
        constraints.update(self.energy_delivered_constraints(rates, active_evs, evse_indexes))
        constraints.update(self.infrastructure_constraints(rates, evse_indexes))

        objective = cp.Maximize(self.obj(rates, active_evs))
        return cp.Problem(objective, list(constraints.values())), constraints, rates

    def _solve(self, active_evs, offset_time, verbose=False, solver=None):
        prob, constraints, rates = self._build_problem(active_evs, offset_time)
        try:
            _ = prob.solve(verbose=verbose, solver=solver)
        except:
            _ = prob.solve(verbose=verbose, solver=cp.SCS)

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError('Problem Infeasible.')

        active_evses = set(ev.station_id for ev in active_evs)
        network_constraints = self.interface.get_constraints()
        evse_indexes = [evse_id for evse_id in network_constraints.evse_index if evse_id in active_evses]
        return {evse_id: np.clip(rates[j, :].value,
                                 a_min=self.interface.min_pilot_signal(evse_id),
                                 a_max=self.interface.max_pilot_signal(evse_id)) for j, evse_id in enumerate(evse_indexes)}

    def schedule(self, active_evs):
        if self.offline:
            if self.internal_schedule is None:
                self.internal_schedule = self._solve(self.evs, 0, solver=self.solver)
            t = self.interface.current_time
            return {ev.station_id: [self.internal_schedule[ev.station_id][t]] for ev in active_evs}
        else:
            if len(active_evs) == 0:
                return {}
            else:
                t = self.interface.current_time
                intermediate_schedule = self._solve(active_evs, t, solver=self.solver)
                if self.post_processor is not None:
                    return self.post_processor.process(intermediate_schedule, active_evs)
                else:
                    return intermediate_schedule


class AdaCharge(AdaChargeBase):
    def __init__(self, const_type=SOC, energy_equality=False, solver=None, max_recomp=None, offline=False, events=None,
                 post_processor=None, regularizers=None):
        obj_config = [(1, quick_charge), (1e-6, equal_share)]
        if regularizers is not None:
            obj_config.extend(regularizers)
        super().__init__(obj_config,
                         const_type, energy_equality, solver, max_recomp, offline, events, post_processor)


class AdaChargeProfitMax(AdaChargeBase):
    def __init__(self, revenue, const_type=SOC, energy_equality=False, solver=None, max_recomp=None, offline=False, events=None,
                 post_processor=None, get_dc=None, regularizers=None):
        """

        Args:
            revenue: $/kWh
            const_type: SOC or AFFINE
            energy_equality: True of False
            solver: any CVXPy solver
            max_recomp: int
            get_dc: function to get the demand charge proxy
        """
        obj_config = [(revenue, quick_charge), (1, energy_cost)]
        if get_dc is not None:
            obj_config.append((1, get_dc))
        else:
            obj_config.append((1, demand_charge))
        if regularizers is not None:
            obj_config.extend(regularizers)
        super().__init__(obj_config, const_type, energy_equality, solver, max_recomp, offline, events, post_processor)


class AdaChargeLoadFlattening(AdaChargeBase):
    def __init__(self, external_signal=None, const_type=SOC, energy_equality=True, solver=None, max_recomp=None,
                 offline=False, events=None, post_processor=None, regularizers=None):
        """

        Args:
            external_signal: np.ndarray of an external signal which we will attempt to flatten.
                Should be at least as long as the simulation.
            const_type: SOC or AFFINE
            energy_equality: True of False
            solver: any CVXPy solver
            max_recomp: int
        """
        obj_config = [(1, load_flattening, {'external_signal': external_signal})]
        if regularizers is not None:
            obj_config.extend(regularizers)
        super().__init__(obj_config, const_type, energy_equality, solver, max_recomp, offline, events, post_processor)
