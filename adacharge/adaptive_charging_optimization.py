from typing import List, Union
from collections import namedtuple
import numpy as np
import cvxpy as cp
from datatypes import SessionInfo, InfrastructureInfo


class InfeasibilityException(Exception):
    pass


ObjectiveComponent = namedtuple('ObjectiveComponent', ['function', 'coefficient', 'kwargs'], defaults=[1, {}])


class AdaptiveChargingOptimization:
    """ Base class for all MPC based charging algorithms.

    Args:
        constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order Cone
            or 'LINEAR' for linearized constraints.
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
    """
    def __init__(self, objective: List[ObjectiveComponent], period, constraint_type='SOC',
                 enforce_energy_equality=False, solver='ECOS'):
        self.period = period
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.objective_configuration = objective

    @staticmethod
    def charging_rate_bounds(rates: cp.Variable, active_sessions: List[SessionInfo], evse_index: List[str]):
        """ Get upper and lower bound constraints for each charging rate.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            evse_index (List[str]): List of IDs for all EVSEs. Index in evse_index represents the row number of that
                EVSE in rates.

        Returns:
            List[cp.Constraint]: List of lower bound constraint, upper bound constraint.
        """
        lb, ub = np.zeros(rates.shape), np.zeros(rates.shape)
        for session in active_sessions:
            i = evse_index.index(session.station_id)
            lb[i, session.arrival_offset:session.arrival_offset + session.remaining_time] = session.min_rates
            ub[i, session.arrival_offset:session.arrival_offset + session.remaining_time] = session.max_rates
        # To ensure feasibility, replace upper bound with lower bound when they conflict
        ub[ub < lb] = lb[ub < lb]
        return [rates >= lb, rates <= ub]

    @staticmethod
    def energy_constraints(rates: cp.Variable, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo,
                           period, enforce_energy_equality=False):
        """ Get constraints on the energy delivered for each session.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            evse_index (List[str]): List of IDs for all EVSEs. Index in evse_index represents the row number of that
                EVSE in rates.
            enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
                If False, energy delivered must be less than or equal to request.

        Returns:
            List[cp.Constraint]: List of energy delivered constraints for each session.
        """
        constraints = []
        for session in active_sessions:
            i = infrastructure.get_station_index(session.station_id)
            planned_energy = cp.sum(rates[i, session.arrival_offset:session.arrival_offset + session.remaining_time])
            planned_energy *= infrastructure.voltages[i] * period / 1e3 / 60
            if enforce_energy_equality:
                constraints.append(planned_energy == session.remaining_energy)
            else:
                constraints.append(planned_energy <= session.remaining_energy)
        return constraints

    @staticmethod
    def infrastructure_constraints(rates: cp.Variable, infrastructure: InfrastructureInfo, constraint_type='SOC'):
        """ Get constraints enforcing infrastructure limits.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order
                Cone or 'LINEAR' for linearized constraints.

        Returns:
            List[cp.Constraint]: List of constraints, one for each bottleneck in the electrical infrastructure.
        """
        constraints = []
        if constraint_type == 'SOC':
            if infrastructure.phases is None:
                raise ValueError('phases is required when using SOC infrastructure constraints.')
            phase_in_rad = np.deg2rad(infrastructure.phases)
            for j, v in enumerate(infrastructure.constraint_matrix):
                a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
                constraints.append(cp.norm(a @ rates, axis=0) <= infrastructure.constraint_limits[j])
        elif constraint_type == 'LINEAR':
            for j, v in enumerate(infrastructure.constraint_matrix):
                constraints.append(np.abs(v) @ rates <= infrastructure.constraint_limits[j])
        else:
            raise ValueError(
                'Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.'.format(constraint_type))
        return constraints

    @staticmethod
    def peak_constraint(rates: cp.Variable, peak_limit: Union[float, List[float], np.ndarray]):
        """ Get constraints enforcing infrastructure limits.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.

        Returns:
            List[cp.Constraint]: List of constraints, one for each bottleneck in the electrical infrastructure.
        """
        if peak_limit is not None:
            return [cp.sum(rates, axis=0) <= peak_limit]
        return []

    def build_objective(self, rates: cp.Variable, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo,
                        current_time: int):
        def _merge_dicts(default_args, user_args):
            """ Merge two dictionaries where user args override default args when their is a conflict. """
            kwargs = dict()
            kwargs.update(default_args)
            kwargs.update(user_args)
            return kwargs

        env_args = {'period': self.period, 'current_time': current_time, 'active_sessions': active_sessions}
        obj = cp.Constant(0)
        for component in self.objective_configuration:
            obj += component.coefficient * component.function(rates, infrastructure,
                                                              **_merge_dicts(env_args, component.kwargs))
        return obj

    def build_problem(self, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo, current_time: int,
                      peak_limit: Union[float, List[float], np.ndarray] = None):
        """ Build parts of the optimization problem including variables, constraints, and objective function.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.

        Returns:
            Dict[str: object]:
                'objective' : cvxpy expression for the objective of the optimization problem
                'constraints': list of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        optimization_horizon = max(s.arrival_offset + s.remaining_time for s in active_sessions)
        num_evses = len(infrastructure.evse_index)
        rates = cp.Variable(shape=(num_evses, optimization_horizon))

        constraints = []

        # Rate constraints
        constraints.extend(self.charging_rate_bounds(rates, active_sessions, infrastructure.evse_index))

        # Energy Delivered Constraints
        constraints.extend(self.energy_constraints(rates, active_sessions, infrastructure,
                                                   self.period, self.enforce_energy_equality))

        # Infrastructure Constraints
        constraints.extend(self.infrastructure_constraints(rates, infrastructure, self.constraint_type))

        # Peak Limit
        constraints.extend(self.peak_constraint(rates, peak_limit))

        # Objective Function
        objective = cp.Maximize(self.build_objective(rates, active_sessions, infrastructure, current_time))
        return {'objective': objective,
                'constraints': constraints,
                'variables': {'rates': rates}}

    def solve(self, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo,
              current_time: int = 0, peak_limit: Union[float, List[float], np.ndarray] = None):
        """ Solve optimization problem to create a schedule of charging rates.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            current_time (int): Time when the optimizer is called. Useful for syncing with prices and external signals.
                Defaults to 0. (periods)
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.


        Returns:
            np.Array: Numpy array of charging rates of shape (N, T) where N is the number of EVSEs in the network and
                T is the length of the optimization horizon. Rows are ordered according to the order of evse_index in
                infrastructure.
        """
        # Here we take in arguments which describe the problem and build a problem instance.
        problem_dict = self.build_problem(active_sessions, infrastructure, current_time, peak_limit)
        prob = cp.Problem(problem_dict['objective'], problem_dict['constraints'])
        prob.solve(solver=self.solver)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise InfeasibilityException(f'Solve failed with status {prob.status}')
        return problem_dict['variables']['rates'].value


# Objective Functions

"""
Inputs: 
    rates -> cp.Variable()
    
Sometime needed
    infrastructure 
    active_sessions
    period
    
Signals -> signals_interface?
    previous_rates
    energy_tariffs 
    demand_charge
    external_demand
    signal_to_follow
"""

# ---------------------------------------------------------------------------------
#  Objective Functions
#
#
#  All objectives should take rates as their first positional argument.
#  All other arguments should be passed as keyword arguments.
#  All functions should except **kwargs as their last argument to avoid errors
#  when unknown arguments are passed.
#
# ---------------------------------------------------------------------------------


def charging_power(rates, infrastructure, **kwargs):
    """ Returns a matrix with the same shape as rates but with units kW instead of A. """
    voltage_matrix = np.tile(infrastructure.voltages, (rates.shape[1], 1)).T
    return cp.multiply(rates, voltage_matrix) / 1e3


def aggregate_power(rates, infrastructure, **kwargs):
    """ Returns aggregate charging power for each time period. """
    return cp.sum(charging_power(rates, infrastructure=infrastructure), axis=0)


def get_period_energy(rates, infrastructure, period, **kwargs):
    """ Return energy delivered in kWh during each time period and each session. """
    power = charging_power(rates, infrastructure=infrastructure)
    period_in_hours = period / 60
    return power * period_in_hours


def aggregate_period_energy(rates, infrastructure, period, **kwargs):
    """ Returns the aggregate energy delivered in kWh during each time period. """
    # get charging rates in kWh per period
    energy_per_period = get_period_energy(rates, infrastructure=infrastructure, period=period)
    return cp.sum(energy_per_period, axis=0)


def quick_charge(rates, infrastructure, **kwargs):
    optimization_horizon = rates.shape[1]
    c = np.array([(optimization_horizon - t) / optimization_horizon for t in range(optimization_horizon)])
    return c @ cp.sum(rates, axis=0)


def equal_share(rates, infrastructure, **kwargs):
    return -cp.sum_squares(rates)


def tou_energy_cost(rates, infrastructure, period, energy_prices, current_time, **kwargs):
    current_prices = energy_prices[current_time: current_time + rates.shape[1]]
    return -current_prices @ aggregate_period_energy(rates, infrastructure, period)


def total_energy(rates, infrastructure, period, **kwargs):
    return cp.sum(get_period_energy(rates, infrastructure, period))


def peak(rates, infrastructure, baseline_peak=0, **kwargs):
    agg_power = aggregate_power(rates, infrastructure)
    max_power = cp.max(agg_power)
    if baseline_peak > 0:
        return cp.maximum(max_power, baseline_peak)
    else:
        return max_power


def load_flattening(rates, infrastructure, external_signal=None, **kwargs):
    if external_signal is None:
        external_signal = np.zeros(rates.shape[1])
    aggregate_rates_kW = aggregate_power(rates, infrastructure)
    total_aggregate = aggregate_rates_kW + external_signal
    return -cp.sum_squares(total_aggregate)


# def smoothing(rates, active_sessions, infrastructure, previous_rates, normp=1, *args, **kwargs):
#     reg = -cp.norm(cp.diff(rates, axis=1), p=normp)
#     prev_mask = np.logical_not(np.isnan(previous_rates))
#     if np.any(prev_mask):
#         reg -= cp.norm(rates[0, prev_mask] - previous_rates[prev_mask], p=normp)
#     return reg

