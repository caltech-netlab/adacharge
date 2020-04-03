from acnportal.algorithms import BaseAlgorithm, least_laxity_first
from copy import deepcopy
import warnings

from adaptive_charging_optimization import *
from preprocessing import apply_rampdown, apply_minimum_charging_rate, enforce_evse_pilot_limit
from postprocessing import project_into_continuous_feasible_pilots, project_into_discrete_feasible_pilots
from postprocessing import index_based_reallocation


def get_infrastructure_info(interface) -> InfrastructureInfo:
    """ Returns an InfrastructureInfo object generated from interface.

    Args:
        interface: An Interface like object. See acnsim.

    Returns:
        InfrastructureInfo: A description of the charging infrastructure.
    """
    constraints = interface.get_constraints()
    constraint_matrix = constraints.constraint_matrix if constraints.constraint_matrix is not None else np.array([])
    magnitudes = constraints.magnitudes if constraints.magnitudes is not None else np.array([])
    phases = np.array([interface.evse_phase(station_id) for station_id in constraints.evse_index])
    voltages = np.array([interface.evse_voltage(station_id) for station_id in constraints.evse_index])
    min_pilot_signals = np.array([interface.min_pilot_signal(station_id) for station_id in constraints.evse_index])
    max_pilot_signals = np.array([interface.max_pilot_signal(station_id) for station_id in constraints.evse_index])
    allowable_rates = [interface.allowable_pilot_signals(station_id)[1] for station_id in constraints.evse_index]
    return InfrastructureInfo(constraint_matrix, magnitudes, phases, voltages,
                              constraints.constraint_index, constraints.evse_index,
                              max_pilot_signals, min_pilot_signals, allowable_rates)


def get_active_sessions(active_evs, interface):
    """ Return a list of SessionInfo objects describing the currently charging EVs.

    Args:
        active_evs (List[acnsim.EV]: List of EV objects from acnsim.
        interface: An interface like object. See acnsim.

    Returns:
        List[SessionInfo]: List of currently active charging sessions.
    """
    return [SessionInfo(ev.station_id, ev.session_id, ev.requested_energy, ev.energy_delivered, ev.arrival,
                        ev.departure, interface.evse_voltage(ev.station_id), interface.current_time)
            for ev in active_evs]


class AdaptiveSchedulingAlgorithm(BaseAlgorithm):
    def __init__(self, objective, constraint_type='SOC', enforce_energy_equality=False, solver=None, peak_limit=None,
                 rampdown=False, minimum_charge=False, quantize=False, reallocate=False,
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
        active_sessions = get_active_sessions(active_evs, self.interface)
        active_sessions = enforce_evse_pilot_limit(active_sessions, infrastructure)
        if self.rampdown is not None:
            active_sessions = apply_rampdown(self.rampdown, active_sessions)
        if self.minimum_charge:
            active_sessions = apply_minimum_charging_rate(active_sessions, infrastructure)
        self.optimizer = AdaptiveChargingOptimization(self.objective, self.interface.period, self.constraint_type,
                                                      self.enforce_energy_equality, solver=self.solver)
        rates_matrix = self.optimizer.solve(active_sessions, infrastructure, self.peak_limit)
        if self.quantize:
            rates_matrix = project_into_discrete_feasible_pilots(rates_matrix, infrastructure)
            if self.reallocate:
                target_peak = rates_matrix[:, 0].sum()
                index_fn = lambda x: x.remaining_time
                rates_matrix[:, 0] = index_based_reallocation(active_sessions, infrastructure, rates_matrix[:, 0],
                                                              target_peak, index_fn)
        else:
            rates_matrix = project_into_continuous_feasible_pilots(rates_matrix, infrastructure)
        rates_matrix = np.maximum(rates_matrix, 0)
        return {station_id: rates_matrix[i, :] for i, station_id in enumerate(infrastructure.evse_index)}


def adacharge_qc(const_type='SOC', energy_equality=False, solver=None, peak_limit=None,
                 regularizers=None, rampdown=None, minimum_charge=False, quantize=False, reallocate=False,
                 max_recompute=None, base=AdaptiveSchedulingAlgorithm):
    """

    Args:
        const_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order Cone
            or 'LINEAR' for linearized constraints.
        energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
        peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
            enforced.
        regularizers (List[ObjectiveComponent]): List of ObjectiveComponents to add to the default objective.
        rampdown (Rampdown): Rampdown object used to predict the maximum charging rate of the EV's battery. If None,
            no ramp down is applied.
        minimum_charge (bool): If true EV should charge at least at the minimum non-zero charging rate of the EVSE
            it is connected to for the first control period.
        quantize (bool): If true, apply project_into_discrete_feasible_pilots post-processing step.
        reallocate (bool): If true, apply index_based_reallocation post-processing step.
        max_recompute (int): Maximum number of control periods between optimization solves.
        base: AdaptiveSchedulingAlgorithm or subclass.

    Returns:
        AdaptiveSchedulingAlgorithm: Scheduling algorithm instance.
    """
    obj_config = [ObjectiveComponent(quick_charge), ObjectiveComponent(equal_share, 1e-12)]
    if regularizers is not None:
        obj_config.extend(regularizers)
    return base(obj_config, const_type, energy_equality, solver, peak_limit, rampdown, minimum_charge, quantize, reallocate)


# def adacharge_profit_max(revenue, const_type='SOC', energy_equality=False, solver=None, peak_limit=None,
#                          regularizers=None, rampdown=None, minimum_charge=False, quantize=False, reallocate=False,
#                          base=AdaptiveSchedulingAlgorithm):
#     """
#
#     Args:
#         revenue: $/kWh
#         const_type: SOC or AFFINE
#         energy_equality: True of False
#         solver: any CVXPy solver
#         max_recomp: int
#         get_dc: function to get the demand charge proxy
#     """
#     obj_config = [(revenue, total_energy), (1, energy_cost)]
#     if get_dc is not None:
#         obj_config.append((1, get_dc))
#     else:
#         obj_config.append((1, demand_charge))
#     if regularizers is not None:
#         obj_config.extend(regularizers)
#     return base(obj_config, const_type, energy_equality, solver, peak_limit, rampdown, minimum_charge, quantize, reallocate)
#
#
# def adacharge_load_flattening(external_signal=None, const_type='SOC', energy_equality=False, solver=None,
#                               peak_limit=None, regularizers=None, rampdown=None, minimum_charge=False, quantize=False,
#                               reallocate=False, base=AdaptiveSchedulingAlgorithm):
#     """
#
#     Args:
#         external_signal: np.ndarray of an external signal which we will attempt to flatten.
#             Should be at least as long as the simulation.
#         const_type: SOC or AFFINE
#         energy_equality: True of False
#         solver: any CVXPy solver
#         max_recomp: int
#     """
#     obj_config = [(1, load_flattening, {'external_signal': external_signal})]
#     if regularizers is not None:
#         obj_config.extend(regularizers)
#     return base(obj_config, const_type, energy_equality, solver, max_recomp, offline, events, post_processor, rampdown, peak_limit)


# # Aliases to not break existing code.
AdaCharge = adacharge_qc
# AdaChargeProfitMax = adacharge_profit_max
# AdaChargeLoadFlattening = adacharge_load_flattening
