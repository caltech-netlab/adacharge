from .cvx_utils import *
import heapq
from acnportal import algorithms


def project_into_set(x, allowable_set, eps=0.05):
    """ Project scalar x into the allowable set.

    If x is within eps of the next value in the set round up, otherwise round down.

    Args:
        x (number): scalar to be projected into the set
        allowable_set (list): List of allowable values
        eps (float): Range to allow value to round up.

    Returns:
        float: x rounded into the allowable_set.
    """
    less_than_x = [s for s in sorted(allowable_set + [0]) if s <= x + eps]
    return less_than_x[-1] if len(less_than_x) > 0 else 0


class IndexPostProcessor:
    def __init__(self, min_charge_rate=True):
        self._interface = None
        self.min_charge_rate = min_charge_rate

    @property
    def interface(self):
        """ Return the algorithm's interface with the environment.

        Returns:
            Interface: An interface to the enviroment.

        Raises:
            ValueError: Exception raised if interface is accessed prior to an interface being registered.
        """
        if self._interface is not None:
            return self._interface
        else:
            raise ValueError('No interface has been registered yet. Please call register_interface prior to using the'
                             'algorithm.')

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
        self._interface = interface

    @staticmethod
    def metric(schedule, target, ev):
        return -(target[ev.station_id][0] - schedule[ev.station_id][0])**2

    def initial_allocation(self, target, active_evs):
        schedule = {}
        allowable_rates = {}
        for evse_id in target:
            _continuous, _allowable_rates = self.interface.allowable_pilot_signals(evse_id)
            if _continuous:
                raise ValueError('Indexed Projection is only designed for discrete allowable rate sets.')
            schedule[evse_id] = [project_into_set(target[evse_id][0], _allowable_rates, eps=1e-3)]
            allowable_rates[evse_id] = _allowable_rates
        rate_idx_map = {evse_id: allowable_rates[evse_id].index(schedule[evse_id][0]) for evse_id in target}
        return schedule, rate_idx_map, allowable_rates

    def process(self, target, active_evs):
        target_peak = sum(t[0] for t in target.values())
        schedule, rate_idx_map, allowable_rates = self.initial_allocation(target, active_evs)
        evs = {ev.station_id: ev for ev in active_evs}
        # put all EVs with non-zero in a heapq sorted by deviation from target
        queue = [(self.metric(schedule, target, ev), ev.station_id)
                 for ev in active_evs if not self.min_charge_rate or schedule[ev.station_id][0] > 0]
        heapq.heapify(queue)
        while len(queue) > 0:
            dev, station_id = heapq.heappop(queue)
            if rate_idx_map[station_id] < len(allowable_rates[station_id]) - 1:
                schedule[station_id][0] = allowable_rates[station_id][rate_idx_map[station_id] + 1]
                new_dev = self.metric(schedule, target, evs[station_id])
                if new_dev > dev and \
                        sum(s[0] for s in schedule.values()) <= target_peak and \
                        self.interface.is_feasible(schedule) and \
                        schedule[station_id][0] <= self.interface.remaining_amp_periods(evs[station_id]):
                    rate_idx_map[station_id] += 1
                    heapq.heappush(queue, (new_dev, station_id))
                else:
                    schedule[station_id][0] = allowable_rates[station_id][rate_idx_map[station_id]]
        return schedule
    
    
class RRPostProcessor(IndexPostProcessor):
        def process(self, target, active_evs):
            target_peak = sum(t[0] for t in target.values())
            schedule, rate_idx_map, allowable_rates = self.initial_allocation(target, active_evs)
            evs = {ev.station_id: ev for ev in active_evs}
            rr = algorithms.RoundRobin(algorithms.least_laxity_first, peak_limit=target_peak)
            rr.register_interface(self.interface)
            schedule = rr.schedule(active_evs, init_schedule=schedule)
            return schedule


class AdaChargePostProcessor:
    def __init__(self, const_type=SOC, solver=None, integer_program=False, round_up_target=False):
        self._interface = None
        self.const_type = const_type
        self.solver = solver
        self.integer_program = integer_program
        self.round_up_target = round_up_target

    @property
    def interface(self):
        """ Return the algorithm's interface with the environment.

        Returns:
            Interface: An interface to the enviroment.

        Raises:
            ValueError: Exception raised if interface is accessed prior to an interface being registered.
        """
        if self._interface is not None:
            return self._interface
        else:
            raise ValueError('No interface has been registered yet. Please call register_interface prior to using the'
                             'algorithm.')

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
        self._interface = interface

    def obj(self, rates, target):
        return -cp.sum_squares(rates - target)

    def feasible_min_rates(self, active_evs, target):
        ev_queue = [ev for ev in sorted(active_evs, key=lambda x: target[x.station_id][0], reverse=True)]
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

        if self.round_up_target:
            target_vector = np.array([np.ceil(target[evse_id][0]) for evse_id in evse_indexes])
        else:
            target_vector = np.array([target[evse_id][0] for evse_id in evse_indexes])
        target_peak = np.sum(target_vector)
        rates = cp.Variable(target_vector.shape, name='rates', integer=self.integer_program)
        constraints = {}

        feasible_min = self.feasible_min_rates(active_evs, target)
        rates_ub = np.zeros(rates.shape)
        rates_lb = np.zeros(rates.shape)
        for ev in active_evs:
            i = evse_indexes.index(ev.station_id)
            rates_lb[i] = feasible_min[ev.station_id]
            rates_ub[i] = min([self.interface.max_pilot_signal(ev.station_id),
                               self.interface.remaining_amp_periods(ev)])
            rates_ub[i] = max(rates_ub[i], rates_lb[i])  # upper bound should never be less than the lower bound

        constraints['Rate Upper Bounds'] = rates <= rates_ub
        constraints['Rate Lower Bounds'] = rates >= rates_lb

        phases = {evse_id: self.interface.evse_phase(evse_id) for evse_id in evse_indexes}
        constraints.update(infrastructure_constraints(rates, network_constraints, evse_indexes, self.const_type, phases))
        constraints['Peak Limit'] = cp.sum(rates) <= target_peak
        objective = cp.Maximize(self.obj(rates, target_vector))
        prob = cp.Problem(objective, list(constraints.values()))

        try:
            _ = prob.solve(solver=self.solver)
        except:
            print('Solve failed. Trying with SCS.')
            _ = prob.solve(solver=cp.SCS)

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
