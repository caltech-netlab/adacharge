from acnportal.algorithms import BaseAlgorithm, least_laxity_first
from copy import deepcopy

from cvx_utils import *
from post_processor import project_into_set


class AdaChargeBase(BaseAlgorithm):
    def __init__(self, obj_config, const_type=SOC, energy_equality=False, solver=None, max_recomp=None, offline=False,
                 events=None, post_processor=None, rampdown=None, minimum_charge=False, peak_limit=None):
        super().__init__(rampdown)
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
                    if event[1].type == 'Plugin' or event[1].type == 'Arrival':
                        self.evs.append(deepcopy(event[1].ev))
        else:
            self.max_recompute = max_recomp
        self.const_type = const_type
        self.energy_equality = energy_equality
        self.solver = solver
        self.post_processor = post_processor
        if post_processor is not None:
            self.max_recompute = 1
        self.minimum_charge = minimum_charge
        self.peak_limit=peak_limit

    def register_interface(self, interface):
        """ Register interface to the _simulator/physical system.

        This interface is the only connection between the algorithm and what it is controlling. Its purpose is to
        abstract the underlying network so that the same algorithms can run on a simulated environment or a physical
        one.

        Args:
            interface (Interface): An interface to the underlying network whether simulated or real.

        Returns:
            None
        """
        super().register_interface(interface)
        if self.post_processor is not None:
            self.post_processor.register_interface(interface)

    def obj(self, rates, active_evs):
        return sum(x[0]*x[1](rates, active_evs, self.interface) if len(x) == 2 else
                   x[0]*x[1](rates, active_evs, self.interface, **x[2]) for x in self.obj_config)

    def individual_rate_constraints(self, rates, active_evs, evse_indexes, minimum_charge=None):
        min_pilots = {ev.session_id: np.zeros(ev.departure - ev.arrival) for ev in active_evs}
        if self.minimum_charge:
            feasible_minimums, removed = self.feasible_min_rates(active_evs)
            for ev in active_evs:
                min_pilots[ev.session_id][0] = feasible_minimums[ev.session_id]

        max_pilots = {ev.session_id: np.repeat(self.interface.max_pilot_signal(ev.station_id), ev.departure - ev.arrival) for ev in active_evs}
        # If rampdown is used, update the maximum pilot to be the min of rampdown and the EVSE max
        if self.rampdown is not None:
            rampdown_pilots = self.rampdown.get_maximum_rates(active_evs)
            for ev in active_evs:
                if ev.session_id in rampdown_pilots:
                    max_pilots[ev.session_id] = np.minimum(max_pilots[ev.session_id], rampdown_pilots[ev.session_id])
                if self.minimum_charge and ev.session_id in removed:
                    max_pilots[ev.session_id][0] = 0

        for ev in active_evs:
            max_pilots[ev.session_id] = np.maximum(max_pilots[ev.session_id], min_pilots[ev.session_id])
        return rate_constraints(rates, active_evs, evse_indexes, max_pilots, min_pilots)

    def energy_delivered_constraints(self, rates, active_evs, evse_indexes):
        remaining_demands = {ev.session_id: self.interface.remaining_amp_periods(ev) for ev in active_evs}
        return energy_constraints(rates, active_evs, evse_indexes, remaining_demands, self.energy_equality)

    def infrastructure_constraints(self, rates, evse_indexes):
        network_constraints = self.interface.get_constraints()
        phases = {evse_id: self.interface.evse_phase(evse_id) for evse_id in evse_indexes}
        return infrastructure_constraints(rates, network_constraints, evse_indexes, self.const_type, phases)

    def feasible_min_rates(self, active_evs):
        ev_queue = least_laxity_first(active_evs, self.interface)
        schedule = {ev.station_id: [0] for ev in ev_queue}
        removed = set()
        for ev in ev_queue:
            continuous, allowable_rates = self.interface.allowable_pilot_signals(ev.station_id)
            schedule[ev.station_id][0] = allowable_rates[0] if continuous else allowable_rates[1]
            if not self.interface.is_feasible(schedule):
                schedule[ev.station_id][0] = 0
                removed.add(ev.session_id)
        return {ev.session_id: schedule[ev.station_id][0] for ev in active_evs}, removed

    def _build_problem(self, active_evs, offset_time):
        network_constraints = self.interface.get_constraints()
        active_evses = set(ev.station_id for ev in active_evs)
        evse_indexes = [evse_id for evse_id in network_constraints.evse_index if evse_id in active_evses]
        for ev in active_evs:
            ev.arrival = max(0, ev.arrival - offset_time)
            ev.departure -= offset_time
        max_t = max(ev.departure for ev in active_evs) + 1

        constraints = {}
        rates = cp.Variable((len(evse_indexes), max_t), name='rates')
        constraints.update(self.individual_rate_constraints(rates, active_evs, evse_indexes))
        constraints.update(self.energy_delivered_constraints(rates, active_evs, evse_indexes))
        constraints.update(self.infrastructure_constraints(rates, evse_indexes))
        if self.peak_limit is not None:
            constraints['peak_limit'] = cp.max(cp.sum(rates, axis=0)) <= self.peak_limit
        objective = cp.Maximize(self.obj(rates, active_evs))
        return cp.Problem(objective, list(constraints.values())), constraints, rates

    def _solve(self, active_evs, offset_time, verbose=False, solver=None):
        if len(active_evs) == 0:
            return {}
        prob, constraints, rates = self._build_problem(active_evs, offset_time)
        try:
            _ = prob.solve(verbose=verbose, solver=solver)
        except:
            print('Solve failed. Trying with SCS.')
            _ = prob.solve(verbose=verbose, solver=cp.SCS)

        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError('Problem Infeasible.')

        active_evses = set(ev.station_id for ev in active_evs)
        network_constraints = self.interface.get_constraints()
        evse_indexes = [evse_id for evse_id in network_constraints.evse_index if evse_id in active_evses]
        schedule = {}
        for j, evse_id in enumerate(evse_indexes):
            schedule[evse_id] = rates[j, :].value
        return schedule

    def project_to_evse_rates(self, intermediate_schedule):
        new_schedule = {}
        for evse_id in intermediate_schedule:
            continuous, allowable_rates = self.interface.allowable_pilot_signals(evse_id)
            if continuous:
                new_schedule[evse_id] = np.clip(intermediate_schedule[evse_id], a_min=0, a_max=allowable_rates[1])
            else:
                new_schedule[evse_id] = [project_into_set(x, allowable_rates) for x in
                                                  intermediate_schedule[evse_id]]
        return new_schedule

    def schedule(self, active_evs):
        if self.minimum_charge:
            # active_evs = self.remove_active_evs_less_than_deadband(active_evs, 6)
            active_evs = self.remove_active_evs_less_than_deadband(active_evs)

        if self.offline:
            if self.internal_schedule is None:
                t = self.interface.current_time
                self.internal_schedule = self._solve(self.evs, 0, solver=self.solver)
            t = self.interface.current_time
            intermediate_schedule = {ev.station_id: [self.internal_schedule[ev.station_id][t]] for ev in active_evs}
            return self.project_to_evse_rates(intermediate_schedule)
        else:
            if len(active_evs) == 0:
                return {}
            else:
                t = self.interface.current_time
                intermediate_schedule = self._solve(active_evs, t, solver=self.solver)
                if self.post_processor is not None:
                    return self.post_processor.process(intermediate_schedule, active_evs)
                else:
                    return self.project_to_evse_rates(intermediate_schedule)


def adacharge_qc(const_type=SOC, energy_equality=False, solver=None, max_recomp=None, offline=False, events=None,
                 post_processor=None, regularizers=None, rampdown=None, minimum_charge=False, base=AdaChargeBase, peak_limit=None):
    obj_config = [(1, quick_charge), (1e-12, equal_share)]
    if regularizers is not None:
        obj_config.extend(regularizers)
    return base(obj_config, const_type, energy_equality, solver, max_recomp, offline, events, post_processor, rampdown,
                minimum_charge, peak_limit)


def adacharge_qc_relu(const_type=SOC, energy_equality=False, solver=None, max_recomp=None, offline=False, events=None,
                 post_processor=None, regularizers=None, rampdown=None, minimum_charge=False, base=AdaChargeBase, peak_limit=None):
    obj_config = [(1, quick_charge_relu), (1e-12, equal_share)]
    if regularizers is not None:
        obj_config.extend(regularizers)
    return base(obj_config, const_type, energy_equality, solver, max_recomp, offline, events, post_processor, rampdown,
                minimum_charge, peak_limit)


def adacharge_profit_max(revenue, const_type=SOC, energy_equality=False, solver=None, max_recomp=None, offline=False,
                         events=None, post_processor=None, regularizers=None, get_dc=None, rampdown=None,
                         minimum_charge=False, base=AdaChargeBase, peak_limit=None):
    """

    Args:
        revenue: $/kWh
        const_type: SOC or AFFINE
        energy_equality: True of False
        solver: any CVXPy solver
        max_recomp: int
        get_dc: function to get the demand charge proxy
    """
    obj_config = [(revenue, total_energy), (1, energy_cost)]
    if get_dc is not None:
        obj_config.append((1, get_dc))
    else:
        obj_config.append((1, demand_charge))
    if regularizers is not None:
        obj_config.extend(regularizers)
    return base(obj_config, const_type, energy_equality, solver, max_recomp, offline, events, post_processor, rampdown, minimum_charge, peak_limit)


def adacharge_load_flattening(external_signal=None, const_type=SOC, energy_equality=True, solver=None, max_recomp=None,
                              offline=False, events=None, post_processor=None, regularizers=None, rampdown=None,
                              base=AdaChargeBase, peak_limit=None):
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
    return base(obj_config, const_type, energy_equality, solver, max_recomp, offline, events, post_processor, rampdown, peak_limit)


# Aliases to not break existing code.
AdaCharge = adacharge_qc
AdaChargeProfitMax = adacharge_profit_max
AdaChargeLoadFlattening = adacharge_load_flattening
