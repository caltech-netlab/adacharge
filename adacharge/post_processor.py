from .cvx_utils import *

J1772_MIN = 6


def project_into_set(x, allowable_set, eps=0.1):
    return [s for s in sorted(allowable_set) if s <= x + eps][-1]


class AdaChargePostProcessor:
    def __init__(self, interface, const_type=SOC, solver=None, eta=1e-5, integer_program=False):
        self.interface = interface
        self.const_type = const_type
        self.solver = solver
        self.eta = eta
        self.integer_program = integer_program

    def obj(self, rates, target, eta):
        return -cp.square(cp.sum(rates) - np.sum(target)) - eta*cp.sum_squares(rates - target)
        # return -cp.sum_squares(rates - target)

    # def _min_allowable_rate(self, ev):
    #     continuous, allowable_rates = self.interface.allowable_pilot_signals(ev.station_id)
    #     return allowable_rates[0] if continuous else allowable_rates[1]

    def feasible_min_rates(self, active_evs, target):
        ev_queue = [ev for ev in sorted(active_evs, key=lambda x: target[x.station_id], reverse=True)]
        schedule = {ev.station_id: [0] for ev in active_evs}
        for ev in ev_queue:
            continuous, allowable_rates = self.interface.allowable_pilot_signals(ev.station_id)
            schedule[ev.station_id][0] = allowable_rates[0] if continuous else allowable_rates[1]
            if not self.interface.is_feasible(schedule):
                schedule[ev.station_id][0] = 0
        return {key: v[0] for key, v in schedule.items()}

    def process(self, target, active_evs):
        if len(active_evs) == 0:
            return {}
        network_constraints = self.interface.get_constraints()
        active_evses = set(ev.station_id for ev in active_evs)
        evse_indexes = [evse_id for evse_id in network_constraints.evse_index if evse_id in active_evses]

        target_vector = np.array([target[evse_id][0] for evse_id in evse_indexes])
        rates = cp.Variable(target_vector.shape, name='rates', integer=self.integer_program)
        constraints = {}

        feasible_min = self.feasible_min_rates(active_evs, target)
        rates_ub = np.zeros(rates.shape)
        rates_lb = np.zeros(rates.shape)
        for ev in active_evs:
            i = evse_indexes.index(ev.station_id)
            rates_lb[i] = feasible_min[ev.station_id]
            rates_ub[i] = min([ self.interface.max_pilot_signal(ev.station_id),
                                self.interface.remaining_amp_periods(ev)])
            rates_ub[i] = max(rates_ub[i], rates_lb[i])  # upper bound should never be less than the lower bound

        constraints['Rate Upper Bounds'] = rates <= rates_ub
        constraints['Rate Lower Bounds'] = rates >= rates_lb

        phases = {evse_id: self.interface.evse_phase(evse_id) for evse_id in evse_indexes}
        constraints.update(infrastructure_constraints(rates, network_constraints, evse_indexes, self.const_type, phases))

        objective = cp.Maximize(self.obj(rates, target_vector, self.eta))
        prob = cp.Problem(objective, list(constraints.values()))

        _ = prob.solve(solver=self.solver)
        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError('Problem Infeasible.')

        active_evses = set(ev.station_id for ev in active_evs)
        network_constraints = self.interface.get_constraints()
        evse_indexes = [evse_id for evse_id in network_constraints.evse_index if evse_id in active_evses]

        schedule = {}
        for j, evse_id in enumerate(evse_indexes):
            # Numerical issues can push the rate outside its allowable range, clip to feasible region
            r = np.clip(rates[j].value, a_min=rates_lb[j], a_max=rates_ub[j])
            continuous, allowable_rates = self.interface.allowable_pilot_signals(evse_id)
            if not continuous:
                r = project_into_set(r, allowable_rates)
            schedule[evse_id] = [r]
        return schedule
