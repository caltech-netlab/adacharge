from acnportal.algorithms import BaseAlgorithm, least_laxity_first
from copy import deepcopy
import warnings

from adaptive_charging_optimization import *
from preprocessing import apply_rampdown, apply_minimum_charging_rate, enforce_evse_pilot_limit
from postprocessing import project_into_continuous_feasible_pilots, project_into_discrete_feasible_pilots
from postprocessing import index_based_reallocation

# ---------------------------------------------------------
#  These utilities translate from Interface format to
#  InfrastructureInfo and SessionInfo formats. This will
#  hopefully be incorporated into a new ACN-Sim Interface
#  in a future release. For now we can use these functions
#  for conversion.
# ---------------------------------------------------------


def get_infrastructure_info(interface) -> InfrastructureInfo:
    """ Returns an InfrastructureInfo object generated from interface.

    Args:
        interface: An Interface like object. See acnsim.

    Returns:
        InfrastructureInfo: A description of the charging infrastructure.
    """
    def fn_to_list(fn, arg_order):
        return np.array([fn(arg) for arg in arg_order])

    constraints = interface.get_constraints()

    # If constraint_matrix of magnitudes is None, replace with empty array
    constraint_matrix = constraints.constraint_matrix if constraints.constraint_matrix is not None else np.array([])
    magnitudes = constraints.magnitudes if constraints.magnitudes is not None else np.array([])

    # Interface gets values one at a time, populate arrays for each field.
    phases = fn_to_list(interface.evse_phase, constraints.evse_index)
    voltages = fn_to_list(interface.evse_voltage, constraints.evse_index)
    min_pilot_signals = fn_to_list(interface.min_pilot_signal, constraints.evse_index)
    max_pilot_signals = fn_to_list(interface.max_pilot_signal, constraints.evse_index)
    allowable_rates = np.array([interface.allowable_pilot_signals(station_id)[1] for station_id in constraints.evse_index])
    return InfrastructureInfo(constraint_matrix, magnitudes, phases, voltages,
                              constraints.constraint_index, constraints.evse_index,
                              max_pilot_signals, min_pilot_signals, allowable_rates)


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
    def __init__(self, objective, constraint_type='SOC', enforce_energy_equality=False, solver=None, peak_limit=None,
                 rampdown=None, minimum_charge=False, quantize=False, reallocate=False,
                 max_recompute=None):
        """ Model Predictive Control based Adaptive Schedule Algorithm compatible with BaseAlgorithm.

        Args:
            objective (List[ObjectiveComponent]): List of ObjectiveComponents for the optimization.
            constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order Cone
                or 'LINEAR' for linearized constraints.
            enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
                If False, energy delivered must be less than or equal to request.
            solver (str): Backend solver to use. See CVXPY for available solvers.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.
            rampdown (Rampdown): Rampdown object used to predict the maximum charging rate of the EV's battery. If None,
                no ramp down is applied.
            minimum_charge (bool): If true EV should charge at least at the minimum non-zero charging rate of the EVSE
                it is connected to for the first control period.
            quantize (bool): If true, apply project_into_discrete_feasible_pilots post-processing step.
            reallocate (bool): If true, apply index_based_reallocation post-processing step.
            max_recompute (int): Maximum number of control periods between optimization solves.
        """
        super().__init__(rampdown)
        self.objective = objective
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.peak_limit = peak_limit
        self.minimum_charge = minimum_charge
        self.quantize = quantize
        self.reallocate = reallocate
        if not self.quantize and self.reallocate:
            raise ValueError('reallocate cannot be true without quantize. Otherwise there is nothing to reallocate :).')
        if self.quantize:
            if self.max_recompute is not None:
                warnings.warn('Overriding max_recompute to 1 since quantization is on.')
            self.max_recompute = 1
        else:
            self.max_recompute = max_recompute

    def schedule(self, active_evs):
        """ See BaseAlgorithm """
        if len(active_evs) == 0:
            return {}
        infrastructure = get_infrastructure_info(self.interface)
        active_sessions = get_active_sessions(active_evs, self.interface.current_time)
        active_sessions = enforce_evse_pilot_limit(active_sessions, infrastructure)
        if self.rampdown is not None:
            active_sessions = apply_rampdown(self.rampdown, active_sessions)
        if self.minimum_charge:
            active_sessions = apply_minimum_charging_rate(active_sessions, infrastructure)
        optimizer = AdaptiveChargingOptimization(self.objective, self.interface.period, self.constraint_type,
                                                 self.enforce_energy_equality, solver=self.solver)
        rates_matrix = optimizer.solve(active_sessions, infrastructure, current_time=self.interface.current_time,
                                       peak_limit=self.peak_limit)
        if self.quantize:
            rates_matrix = project_into_discrete_feasible_pilots(rates_matrix, infrastructure)
            if self.reallocate:
                target_peak = rates_matrix[:, 0].sum()
                index_fn = lambda x: x.remaining_time
                rates_matrix[:, 0] = index_based_reallocation(rates_matrix[:, 0], active_sessions, infrastructure,
                                                              target_peak, index_fn)
        else:
            rates_matrix = project_into_continuous_feasible_pilots(rates_matrix, infrastructure)
        rates_matrix = np.maximum(rates_matrix, 0)
        return {station_id: rates_matrix[i, :] for i, station_id in enumerate(infrastructure.evse_index)}


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

        active_evs = [deepcopy(event[1].ev) for event in events.queue if event[1].type == 'Plugin']
        self.sessions = get_active_sessions(active_evs, 0)
        self.session_ids = set(s.session_id for s in self.sessions)
        self.internal_schedule = None

    def solve(self):
        if self.interface is None:
            raise ValueError('Error: self.interface is None. Please register interface before calling solve.')
        infrastructure = get_infrastructure_info(self.interface)
        self.sessions = enforce_evse_pilot_limit(self.sessions, infrastructure)

        optimizer = AdaptiveChargingOptimization(self.objective, self.interface.period, self.constraint_type,
                                                 self.enforce_energy_equality, solver=self.solver)
        rates_matrix = optimizer.solve(self.sessions, infrastructure, self.peak_limit)
        rates_matrix = project_into_continuous_feasible_pilots(rates_matrix, infrastructure)
        self.internal_schedule = {station_id: rates_matrix[i, :]
                                  for i, station_id in enumerate(infrastructure.evse_index)}

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

