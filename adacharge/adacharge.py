from acnportal.algorithms import BaseAlgorithm, least_laxity_first
from copy import deepcopy
import warnings

from adaptive_charging_optimization import *
from acnportal.algorithms import apply_upper_bound_estimate, \
    apply_minimum_charging_rate, enforce_pilot_limit
from postprocessing import project_into_continuous_feasible_pilots, project_into_discrete_feasible_pilots
from postprocessing import index_based_reallocation, diff_based_reallocation

# ---------------------------------------------------------
#  These utilities translate from Interface format to
#  InfrastructureInfo and SessionInfo formats. This will
#  hopefully be incorporated into a new ACN-Sim Interface
#  in a future release. For now we can use these functions
#  for conversion.
# ---------------------------------------------------------


# def get_infrastructure_info(interface) -> InfrastructureInfo:
#     """ Returns an InfrastructureInfo object generated from interface.
#
#     Args:
#         interface: An Interface like object. See acnsim.
#
#     Returns:
#         InfrastructureInfo: A description of the charging infrastructure.
#     """
#     def fn_to_list(fn, arg_order):
#         return np.array([fn(arg) for arg in arg_order])
#
#     constraints = interface.get_constraints()
#
#     # If constraint_matrix of magnitudes is None, replace with empty array
#     constraint_matrix = constraints.constraint_matrix if constraints.constraint_matrix is not None else np.array([])
#     magnitudes = constraints.magnitudes if constraints.magnitudes is not None else np.array([])
#
#     # Interface gets values one at a time, populate arrays for each field.
#     phases = fn_to_list(interface.evse_phase, constraints.station_ids)
#     voltages = fn_to_list(interface.evse_voltage, constraints.station_ids)
#     min_pilot_signals = fn_to_list(interface.min_pilot_signal, constraints.station_ids)
#     max_pilot_signals = fn_to_list(interface.max_pilot_signal, constraints.station_ids)
#     allowable_rates = np.array([interface.allowable_pilot_signals(station_id)[1] for station_id in constraints.station_ids])
#     return InfrastructureInfo(constraint_matrix, magnitudes, phases, voltages,
#                               constraints.constraint_index, constraints.station_ids,
#                               max_pilot_signals, min_pilot_signals, allowable_rates)


def get_active_sessions(active_evs, current_time):
    """ Return a list of SessionInfo objects describing the currently charging EVs.

    Args:
        active_evs (List[acnsim.EV]: List of EV objects from acnsim.
        current_time (int): Current time of the simulation.

    Returns:
        List[SessionInfo]: List of currently active charging sessions.
    """
    return [SessionInfo(ev.station_id, ev.session_id, ev.requested_energy, ev.energy_delivered, ev.arrival,
                        ev.departure, current_time) for ev in active_evs]


class AdaptiveSchedulingAlgorithm(BaseAlgorithm):
    def __init__(self, objective, constraint_type='SOC',
                 enforce_energy_equality=False, solver=None, peak_limit=None,
                 estimate_max_rate=False, max_rate_estimator=None,
                 uninterrupted_charging=False, quantize=False,
                 reallocate=False, max_recompute=None,
                 allow_overcharging=False):
        """ Model Predictive Control based Adaptive Schedule Algorithm compatible with BaseAlgorithm.

        Args:
            objective (List[ObjectiveComponent]): List of ObjectiveComponents
                for the optimization.
            constraint_type (str): String representing which constraint type
                to use. Options are 'SOC' for Second Order Cone or 'LINEAR'
                for linearized constraints.
            enforce_energy_equality (bool): If True, energy delivered must
                be  equal to energy requested for each EV. If False, energy
                delivered must be less than or equal to request.
            solver (str): Backend solver to use. See CVXPY for available solvers.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on
            aggregate peak current. If None, no limit is enforced.
            rampdown (Rampdown): Rampdown object used to predict the maximum
                charging rate of the EV's battery. If None, no ramp down is
                applied.
            minimum_charge (bool): If true EV should charge at least at the
                minimum non-zero charging rate of the EVSE it is connected
                to for the first control period.
            quantize (bool): If true, apply project_into_discrete_feasible_pilots post-processing step.
            reallocate (bool): If true, apply index_based_reallocation
                post-processing step.
            max_recompute (int): Maximum number of control periods between
                optimization solves.
            allow_overcharging (bool): Allow the algorithm to exceed the energy
                request of the session by at most the energy delivered at the
                minimum allowable rate for one period.
        """
        super().__init__()
        self.objective = objective
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.peak_limit = peak_limit
        self.estimate_max_rate = estimate_max_rate
        self.max_rate_estimator = max_rate_estimator
        self.uninterrupted_charging = uninterrupted_charging
        self.quantize = quantize
        self.reallocate = reallocate
        if not self.quantize and self.reallocate:
            raise ValueError('reallocate cannot be true without quantize. '
                             'Otherwise there is nothing to reallocate :).')
        if self.quantize:
            if self.max_recompute is not None:
                warnings.warn('Overriding max_recompute to 1 '
                              'since quantization is on.')
            self.max_recompute = 1
        else:
            self.max_recompute = max_recompute
        self.allow_overcharging = allow_overcharging

    def register_interface(self, interface):
        """ Register interface to the _simulator/physical system.

        This interface is the only connection between the algorithm and what it
            is controlling. Its purpose is to abstract the underlying
            network so that the same algorithms can run on a simulated
            environment or a physical one.

        Args:
            interface (Interface): An interface to the underlying network
                whether simulated or real.

        Returns:
            None
        """
        self._interface = interface
        if self.max_rate_estimator is not None:
            self.max_rate_estimator.register_interface(interface)

    def schedule(self, active_sessions):
        """ See BaseAlgorithm """
        if len(active_sessions) == 0:
            return {}
        infrastructure = self.interface.infrastructure_info()

        active_sessions = enforce_pilot_limit(active_sessions,
                                                   infrastructure)

        if self.estimate_max_rate:
            active_sessions = apply_upper_bound_estimate(self.max_rate_estimator,
                                             active_sessions)
        if self.uninterrupted_charging:
            active_sessions = apply_minimum_charging_rate(active_sessions,
                                                          infrastructure,
                                                          self.interface)

        optimizer = AdaptiveChargingOptimization(self.objective,
                                                 self.interface,
                                                 self.constraint_type,
                                                 self.enforce_energy_equality,
                                                 solver=self.solver)

        rates_matrix = optimizer.solve(active_sessions,
                                       infrastructure,
                                       peak_limit=self.peak_limit,
                                       prev_peak=self.interface.get_prev_peak())
        if self.quantize:
            if self.reallocate:
                rates_matrix = diff_based_reallocation(rates_matrix,
                                                       active_sessions,
                                                       infrastructure,
                                                       self.interface)
            else:
                rates_matrix = project_into_discrete_feasible_pilots(rates_matrix,
                                                                     infrastructure)
        else:
            rates_matrix = project_into_continuous_feasible_pilots(rates_matrix,
                                                                   infrastructure)
        rates_matrix = np.maximum(rates_matrix, 0)
        return {station_id: rates_matrix[i, :] for i, station_id
                in enumerate(infrastructure.station_ids)}


class AdaptiveChargingAlgorithmOffline(BaseAlgorithm):
    """ Offline optimization for objective with perfect future information.

    The offline optimization assumes ideal EVSEs and batteries. If non-ideal models are used the results are not
    guaranteed to be optimal nor feasible.

    Args:
        objective (List[ObjectiveComponent]): List of ObjectiveComponents for the optimization.
        events (List[Event-like]): List of events which will occur. Only Plugin events are considered.
        constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order Cone
            or 'LINEAR' for linearized constraints.
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
        peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
            enforced.
    """
    def __init__(self, objective, events, constraint_type='SOC', enforce_energy_equality=False, solver=None,
                 peak_limit=None):
        super().__init__()
        self.max_recompute = 1
        self.objective = objective
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.peak_limit = peak_limit

        active_evs = [deepcopy(event[1].ev) for event in events.queue if event[1].event_type == 'Plugin']
        self.sessions = get_active_sessions(active_evs, 0)
        self.session_ids = set(s.session_id for s in self.sessions)
        self.internal_schedule = None

    def solve(self):
        if self.interface is None:
            raise ValueError('Error: self.interface is None. Please register interface before calling solve.')
        infrastructure = self.interface.infrastructure_info()
        self.sessions = enforce_pilot_limit(self.sessions, infrastructure)
        optimizer = AdaptiveChargingOptimization(self.objective,
                                                 self.interface,
                                                 self.constraint_type,
                                                 self.enforce_energy_equality,
                                                 solver=self.solver)
        rates_matrix = optimizer.solve(self.sessions, infrastructure, self.peak_limit)
        rates_matrix = project_into_continuous_feasible_pilots(rates_matrix, infrastructure)
        self.internal_schedule = {station_id: rates_matrix[i, :]
                                  for i, station_id in enumerate(infrastructure.station_ids)}

    def schedule(self, active_evs):
        """ See BaseAlgorithm """
        if self.internal_schedule is None:
            raise ValueError('No internal schedule found. Make sure to call solve before calling schedule or running'
                             'a simulation.')
        for ev in active_evs:
            if ev.session_id not in self.session_ids:
                raise ValueError(f'Error: Session {ev.session_id} not included in offline solve.')
        current_time = self.interface.current_time
        return {ev.station_id: [self.internal_schedule[ev.station_id][current_time]] for ev in active_evs}

