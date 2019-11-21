import numpy as np
from acnportal.algorithms import BaseAlgorithm
import cvxpy as cp
from copy import deepcopy


AFFINE = 'AFFINE'
SOC = 'SOC'


class AdaCharge(BaseAlgorithm):
    def __init__(self, const_type=SOC, energy_equality=False, solver=None, max_recomp=None, offline=False, events=None):
        super().__init__()
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

    def obj(self, rates, active_evs):
        max_t = max(ev.departure for ev in active_evs) + 1
        c = np.array([(max_t - t)/max_t for t in range(max_t)])
        return c*cp.sum(rates, axis=0) - (1e-6/max_t)*cp.sum_squares(rates)

    def _rate_constraints(self, rates, active_evs, evse_indexes):
        constraints = {}
        rates_ub = np.zeros(rates.shape)
        rates_lb = np.zeros(rates.shape)
        for ev in active_evs:
            i = evse_indexes.index(ev.station_id)
            rates_ub[i, ev.arrival:ev.departure] = self.interface.max_pilot_signal(ev.station_id)
        constraints['Rate Upper Bounds'] = rates <= rates_ub
        constraints['Rate Lower Bounds'] = rates >= rates_lb
        return constraints

    def _energy_constraints(self, rates, active_evs, evse_indexes):
        constraints = {}
        for ev in active_evs:
            # Constraint on the energy delivered to each EV
            i = evse_indexes.index(ev.station_id)
            e = cp.Parameter(nonneg=True, name='{0}_energy_request'.format(ev.session_id))
            e.value = self.interface.remaining_amp_periods(ev)
            constraint_id = 'Energy Constraint {0}'.format(ev.session_id)
            if self.energy_equality:
                constraints[constraint_id] = cp.sum(rates[i, ev.arrival:ev.departure]) == e
            else:
                constraints[constraint_id] = cp.sum(rates[i, ev.arrival:ev.departure]) <= e
        return constraints

    @staticmethod
    def _affine_infrastructure_constraints(rates, network_constraints, evse_indexes):
        if network_constraints.constraint_matrix is None:
            return []
        constraints = {}
        trimmed_constraints = network_constraints.constraint_matrix[:, np.isin(network_constraints.evse_index, evse_indexes)]
        inactive_mask = ~np.all(trimmed_constraints == 0, axis=1)
        trimmed_constraints = trimmed_constraints[inactive_mask]
        trimmed_constraint_ids = np.array(network_constraints.constraint_index)[inactive_mask]
        for j in range(trimmed_constraints.shape[0]):
            v = np.abs(trimmed_constraints[j, :])
            constraints[str(trimmed_constraint_ids[j])] = v * rates <= network_constraints.magnitudes[inactive_mask][j]
        return constraints

    def _soc_infrastructure_constraints(self, rates, network_constraints, evse_indexes):
        if network_constraints.constraint_matrix is None:
            return []
        constraints = {}
        trimmed_constraints = network_constraints.constraint_matrix[:, np.isin(network_constraints.evse_index, evse_indexes)]
        inactive_mask = ~np.all(trimmed_constraints == 0, axis=1)
        trimmed_constraints = trimmed_constraints[inactive_mask]
        trimmed_constraint_ids = np.array(network_constraints.constraint_index)[inactive_mask]
        phase_vector = np.array([np.deg2rad(self.interface.evse_phase(evse_id)) for evse_id in evse_indexes])

        for j in range(trimmed_constraints.shape[0]):
            v = np.stack([trimmed_constraints[j, :] * np.cos(phase_vector), trimmed_constraints[j, :] * np.sin(phase_vector)])
            constraints[str(trimmed_constraint_ids[j])] = cp.norm(v * rates, axis=0) <= network_constraints.magnitudes[inactive_mask][j]
        return constraints

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
        constraints.update(self._rate_constraints(rates, active_evs, evse_indexes))
        constraints.update(self._energy_constraints(rates, active_evs, evse_indexes))

        if self.const_type == AFFINE:
            constraints.update(self._affine_infrastructure_constraints(rates, network_constraints, evse_indexes))
        elif self.const_type == SOC:
            constraints.update(self._soc_infrastructure_constraints(rates, network_constraints, evse_indexes))

        objective = cp.Maximize(self.obj(rates, active_evs))
        return cp.Problem(objective, list(constraints.values())), constraints

    def _solve(self, active_evs, offset_time, verbose=False, solver=None):
        prob, constraints = self._build_problem(active_evs, offset_time)
        _ = prob.solve(verbose=verbose, solver=solver)
        if prob.status in ["infeasible", "unbounded"]:
            raise ValueError('Problem Infeasible.')

        rates = next((x for x in prob.variables() if x.name() == 'rates'), None)
        if rates is None:
            raise ValueError('Rates variable not found.')

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
                return self._solve(active_evs, t, solver=self.solver)


class AdaChargeProfitMax(AdaCharge):
    def __init__(self, revenue, const_type=SOC, energy_equality=False, solver=None, max_recomp=None, get_dc=None,
                 offline=False, events=None):
        """

        Args:
            revenue: $/kWh
            const_type: SOC or AFFINE
            energy_equality: True of False
            solver: any CVXPy solver
            max_recomp: int
            get_dc: function to get the demand charge proxy
        """
        super().__init__(const_type, energy_equality, solver, max_recomp, offline, events)
        self.revenue = revenue
        if get_dc is not None:
            self.get_demand_charge = get_dc

    @staticmethod
    def get_demand_charge(iface):
        return iface.get_demand_charge()

    def obj(self, rates, active_evs):
        # TODO(zach): Should account for EVSEs with different voltages
        # We implicitly assume that energy_prices, scaled_revenue, and demand_charge should be scaled by voltage/1000.
        max_t = max(ev.departure for ev in active_evs) + 1
        voltage = self.interface.evse_voltage(active_evs[0].station_id)

        energy_prices = np.array(self.interface.get_prices(max_t)) * (self.interface.period / 60) * voltage / 1000

        scaled_revenue = self.revenue * (self.interface.period / 60) * voltage / 1000

        demand_charge = self.get_demand_charge(self.interface) * voltage / 1000
        schedule_peak = cp.max(cp.sum(rates, axis=0))
        profit = scaled_revenue*cp.sum(rates) - energy_prices*cp.sum(rates, axis=0) - \
                 demand_charge*cp.maximum(schedule_peak, self.interface.get_prev_peak())
        return profit


class AdaChargeLoadFlattening(AdaCharge):
    def __init__(self, external_signal=None, const_type=SOC, energy_equality=True, solver=None, max_recomp=None,
                 offline=False, events=None):
        """

        Args:
            external_signal: np.ndarray of an external signal which we will attempt to flatten.
                Should be at least as long as the simulation.
            const_type: SOC or AFFINE
            energy_equality: True of False
            solver: any CVXPy solver
            max_recomp: int
        """
        super().__init__(const_type, energy_equality, solver, max_recomp, offline, events)
        self.external_signal = external_signal

    def obj(self, rates, active_evs):
        if self.external_signal is None:
            return -cp.sum_squares(cp.sum(rates, axis=0))
        else:
            max_t = max(ev.departure for ev in active_evs) + 1
            t = self.interface.current_time
            voltage = self.interface.evse_voltage(active_evs[0].station_id)
            return -cp.sum_squares(cp.sum(rates, axis=0) - self.external_signal[t: t + max_t] * 1000 / voltage)

