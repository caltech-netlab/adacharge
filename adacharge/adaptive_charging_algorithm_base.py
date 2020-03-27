from typing import List
import numpy as np
import cvxpy as cp
from algo_datatypes import SessionInfo, InfrastructureInfo


class InfeasibilityException(Exception):
    pass


class AdaptiveChargingAlgorithmBase:
    """ Base class for all MPC based charging algorithms.

    Args:
        constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order Cone
            or 'LINEAR' for linearized constraints.
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
        peak_limit (float): Limit on aggregate peak current. If None, no limit is enforced.
    """
    def __init__(self, constraint_type='SOC', enforce_energy_equality=False, solver='ECOS', peak_limit=None):
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.peak_limit = peak_limit

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
    def energy_constraints(rates: cp.Variable, active_sessions: List[SessionInfo], evse_index: List[str], enforce_energy_equality=False):
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
            i = evse_index.index(session.station_id)
            constraint_id = 'EnergyConstraint{0}'.format(session.session_id)
            if enforce_energy_equality:
                constraints.append(cp.sum(rates[i, session.arrival_offset:session.arrival_offset + session.remaining_time]) == session.remaining_energy)
            else:
                constraints.append(cp.sum(rates[i, session.arrival_offset:session.arrival_offset + session.remaining_time]) <= session.remaining_energy)
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
            phase_in_rad = np.deg2rad(infrastructure.phases)
            if infrastructure.phases is None:
                raise ValueError('phases is required when using SOC infrastructure constraints.')
            for j, v in enumerate(infrastructure.constraint_matrix):
                a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
                constraints.append(cp.norm(a @ rates, axis=0) <= infrastructure.magnitudes[j])
        elif constraint_type == 'LINEAR':
            for j, v in enumerate(infrastructure.constraint_matrix):
                constraints.append(np.abs(v) @ rates <= infrastructure.magnitudes[j])
        else:
            raise ValueError(
                'Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.'.format(constraint_type))
        return constraints

    def build_problem(self, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo):
        """ Build parts of the optimization problem including variables, constraints, and objective functin.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.

        Returns:
            Dict[str: object]:
                'objective' : cvxpy expression for the objective of the optimization problem
                'constraints': list of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        # Here we take in arguments which describe the problem and build a problem instance.
        optimization_horizon = max(s.arrival_offset + s.remaining_time for s in active_sessions)
        num_evses = len(infrastructure.evse_index)
        rates = cp.Variable(shape=(num_evses, optimization_horizon))

        constraints = []

        # Rate constraints
        constraints.extend(self.charging_rate_bounds(rates, active_sessions, infrastructure.evse_index))

        # Energy Delivered Constraints
        constraints.extend(self.energy_constraints(rates, active_sessions, infrastructure.evse_index, self.enforce_energy_equality))

        # Infrastructure Constraints
        constraints.extend(self.infrastructure_constraints(rates, infrastructure, self.constraint_type))

        # Peak Limit
        if self.peak_limit is not None:
            constraints.append(cp.max(cp.sum(rates, axis=0)) <= self.peak_limit)

        # Objective Function
        c = np.array([(optimization_horizon - t) / optimization_horizon for t in range(optimization_horizon)])
        obj = c @ cp.sum(rates, axis=0)
        objective = cp.Maximize(obj)
        return {'objective': objective,
                'constraints': constraints,
                'variables': {'rates': rates}}

    def solve(self, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo):
        """ Solve optimization problem to create a schedule of charging rates.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.

        Returns:
            np.Array: Numpy array of charging rates of shape (N, T) where N is the number of EVSEs in the network and
                T is the length of the optimization horizon. Rows are ordered according to the order of evse_index in
                infrastructure.
        """
        problem_dict = self.build_problem(active_sessions, infrastructure)
        prob = cp.Problem(problem_dict['objective'], problem_dict['constraints'])
        prob.solve(solver=self.solver)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise InfeasibilityException(f'Solve failed with status {prob.status}')
        return problem_dict['variables']['rates'].value