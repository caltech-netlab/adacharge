from unittest import TestCase
from unittest.mock import Mock
from adacharge.adaptive_charging_optimization import *
from acnportal.algorithms.tests.generate_test_cases import *
from acnportal.algorithms.tests.testing_interface import TestingInterface
import time

# from testing_utilities import *

DEFAULT_OBJECTIVE = [ObjectiveComponent(quick_charge)]


def mock_interface(period, current_time=0):
    interface = Mock()
    interface.period = period
    interface.current_time = current_time
    return interface


class BaseAlgoTestScenarios(TestCase):
    def setUp(self) -> None:
        self.period = 5  # minutes
        self.max_rate = 32  # A
        self.energy_demand = 3.3  # kW
        self.horizon = 12
        self.current_time = 0
        self.sessions = []
        self.infrastructure = None
        self.rates = None

    def interface(self, sessions_dict, infra_dict):
        return TestingInterface(
            {
                "active_sessions": sessions_dict,
                "infrastructure_info": infra_dict,
                "current_time": self.current_time,
                "period": self.period,
            }
        )

    def build_and_run(self, session_dict, infra_dict, energy_equality=False):
        interface = self.interface(session_dict, infra_dict)
        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()
        asa = AdaptiveChargingOptimization(
            DEFAULT_OBJECTIVE, interface, enforce_energy_equality=energy_equality
        )
        self.rates = asa.solve(self.sessions, self.infrastructure)

    def test_all_rates_less_than_limit(self):
        assert (self.rates <= self.max_rate + 1e-3).all()

    def test_all_energy_demands_met(self):
        energy_expected = np.zeros(self.rates.shape[0])
        energy_delivered = np.zeros(self.rates.shape[0])
        for s in self.sessions:
            i = self.infrastructure.station_ids.index(s.station_id)
            energy_expected[i] = s.remaining_demand
            energy_delivered[i] = self.rates[
                i, s.arrival_offset : s.arrival_offset + s.remaining_time
            ].sum()
            energy_delivered[i] *= (
                self.infrastructure.voltages[i] * self.period / 1e3 / 60
            )
        assert np.allclose(energy_delivered, energy_expected, atol=1e-4, rtol=1e-4)

    def test_no_charging_when_not_plugged_in(self):
        not_plugged_in = np.ones(self.rates.shape, dtype=bool)
        for s in self.sessions:
            i = self.infrastructure.station_ids.index(s.station_id)
            not_plugged_in[
                i, s.arrival_offset : s.arrival_offset + s.remaining_time
            ] = 0
        assert np.allclose(self.rates[not_plugged_in], 0)

    def test_infrastructure_constraints_satisfied(self):
        phase_in_rad = np.deg2rad(self.infrastructure.phases)
        for j, v in enumerate(self.infrastructure.constraint_matrix):
            a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
            line_currents = np.linalg.norm(a @ self.rates, axis=0)
            assert np.all(
                line_currents <= self.infrastructure.constraint_limits[j] + 1e-3
            )


# Simple Correctness Tests
class TestTinyFeasibleNetwork(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        super().setUp()
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[self.horizon] * 2,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        self.build_and_run(sessions_dict, infra_dict)


class TestTinyFeasibleNetworkEnergyEquality(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        super().setUp()
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[self.horizon] * 2,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        self.build_and_run(sessions_dict, infra_dict, True)


class TestTinyInfeasibleBecauseOfMaxRateNetworkEnergyEquality(TestCase):
    def test_infeasible_input_with_equality_constraints(self):
        period = 5
        max_rate = 32
        energy_demand = 3.3
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[12, 4],
            remaining_energy=[energy_demand] * 2,
            requested_energy=[energy_demand] * 2,
            max_rates=[max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        interface = TestingInterface(
            {
                "active_sessions": sessions_dict,
                "infrastructure_info": infra_dict,
                "period": period,
                "current_time": 0,
            }
        )
        asa = AdaptiveChargingOptimization(
            DEFAULT_OBJECTIVE, interface, enforce_energy_equality=True
        )
        with self.assertRaises(InfeasibilityException):
            _ = asa.solve(interface.active_sessions(), interface.infrastructure_info())


class TestTinyInfeasibleBecauseOfInfrastructureNetworkEnergyEquality(TestCase):
    def test_infeasible_input_with_equality_constraints(self):
        period = 5
        max_rate = 32
        energy_demand = 3.3
        horizon = 12
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[horizon] * 2,
            remaining_energy=[energy_demand] * 2,
            requested_energy=[energy_demand] * 2,
            max_rates=[max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=30)
        interface = TestingInterface(
            {
                "active_sessions": sessions_dict,
                "infrastructure_info": infra_dict,
                "period": period,
                "current_time": 0,
            }
        )
        asa = AdaptiveChargingOptimization(
            DEFAULT_OBJECTIVE, interface, enforce_energy_equality=True
        )
        with self.assertRaises(InfeasibilityException):
            _ = asa.solve(interface.active_sessions(), interface.infrastructure_info())


class TestTinyFeasibleNetworkDelayedStart(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        super().setUp()
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0, 4],
            departures=[self.horizon, self.horizon + 4],
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        self.build_and_run(sessions_dict, infra_dict)


class TestTinyFeasibleMultipleSessionsSameEVSE(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        super().setUp()
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0, 12],
            departures=[self.horizon, self.horizon + 12],
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            station_ids=["0"] * 2,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        self.build_and_run(sessions_dict, infra_dict)


class TestTinyMinimumCharge(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        super().setUp()
        self.min_rate = 6
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[self.horizon] * 2,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            min_rates=[self.min_rate] * 2,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        self.build_and_run(sessions_dict, infra_dict)

    def test_all_rates_greater_than_limit(self):
        assert (self.rates >= self.min_rate - 1e-7).all()


class TestTinyPeakLimitScalar(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        super().setUp()
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[self.horizon] * 2,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        interface = self.interface(sessions_dict, infra_dict)

        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()

        asa = AdaptiveChargingOptimization(DEFAULT_OBJECTIVE, interface)
        self.peak_limit = 32
        self.rates = asa.solve(
            self.sessions, self.infrastructure, peak_limit=self.peak_limit
        )

    def test_peak_less_than_limit(self):
        assert (self.rates.sum(axis=0) <= self.peak_limit + 1e-7).all()


class TestTinyPeakLimitVector(TestTinyPeakLimitScalar):
    def setUp(self) -> None:
        super().setUp()
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[self.horizon] * 2,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        interface = self.interface(sessions_dict, infra_dict)

        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()

        asa = AdaptiveChargingOptimization(DEFAULT_OBJECTIVE, interface)
        self.peak_limit = np.array([40] * 6 + [24] * 6)
        self.rates = asa.solve(
            self.sessions, self.infrastructure, peak_limit=self.peak_limit
        )


# Basic Stress Tests
class TestLargeFeasibleSinglePhase(BaseAlgoTestScenarios):
    def setUp(self):
        super().setUp()
        self.energy_demand = 10
        self.horizon = 12 * 12
        N = 54
        energy_demand = [self.energy_demand] * N
        sessions_dict = session_generator(
            num_sessions=N,
            arrivals=[0] * N,
            departures=[self.horizon] * N,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * N,
        )
        infra_dict = single_phase_single_constraint(num_evses=N, limit=32 * N / 3)
        interface = self.interface(sessions_dict, infra_dict)

        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()

        asa = AdaptiveChargingOptimization(
            DEFAULT_OBJECTIVE, interface, constraint_type="LINEAR"
        )

        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infrastructure)
        print(time.time() - start_time)


class TestLargeFeasibleSinglePhaseNetworkSOCConstraints(BaseAlgoTestScenarios):
    def setUp(self):
        super().setUp()
        self.energy_demand = 10
        self.horizon = 12 * 12
        N = 54
        energy_demand = [self.energy_demand] * N
        sessions_dict = session_generator(
            num_sessions=N,
            arrivals=[0] * N,
            departures=[self.horizon] * N,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * N,
        )
        infra_dict = single_phase_single_constraint(num_evses=N, limit=32 * N / 3)
        interface = self.interface(sessions_dict, infra_dict)

        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()

        asa = AdaptiveChargingOptimization(
            DEFAULT_OBJECTIVE, interface, constraint_type="SOC"
        )

        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infrastructure)
        print(time.time() - start_time)


# TODO (zach): It is still unclear why OSQP fails here.
# class TestLargeFeasibleSinglePhaseOSQP(BaseAlgoTestScenarios):
#     def setUp(self):
#         super().setUp()
#         self.energy_demand = 10
#         self.horizon = 12*12
#         N = 54
#         energy_demand = [self.energy_demand] * N
#         sessions_dict = session_generator(num_sessions=N,
#                                           arrivals=[0] * N,
#                                           departures=[self.horizon] * N,
#                                           remaining_energy=energy_demand,
#                                           requested_energy=energy_demand,
#                                           max_rates=[self.max_rate] * N)
#         infra_dict = single_phase_single_constraint(num_evses=N, limit=32*N/3)
#         interface = self.interface(sessions_dict, infra_dict)
#
#         self.infrastructure = interface.infrastructure_info()
#         self.sessions = interface.active_sessions()
#
#         asa = AdaptiveChargingOptimization([DEFAULT_OBJECTIVE[0]], interface,
#                                            constraint_type='LINEAR',
#                                            solver='OSQP')
#         start_time = time.time()
#         self.rates = asa.solve(self.sessions, self.infrastructure)
#         print(time.time() - start_time)


class TestLargeFeasibleThreePhaseSOC(BaseAlgoTestScenarios):
    def setUp(self):
        super().setUp()
        self.energy_demand = 10
        self.horizon = 12 * 12
        N = 54
        energy_demand = [self.energy_demand] * N
        sessions_dict = session_generator(
            num_sessions=N,
            arrivals=[0] * N,
            departures=[self.horizon] * N,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * N,
        )
        infra_dict = three_phase_balanced_network(
            evses_per_phase=N // 3, limit=32 * N / 3
        )
        interface = self.interface(sessions_dict, infra_dict)

        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()

        asa = AdaptiveChargingOptimization(
            DEFAULT_OBJECTIVE, interface, constraint_type="SOC"
        )

        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infrastructure)
        print(time.time() - start_time)


class TestLargeFeasibleThreePhaseWithEqualShareSOC(BaseAlgoTestScenarios):
    def setUp(self):
        super().setUp()
        self.energy_demand = 10
        self.horizon = 12 * 12
        N = 54
        energy_demand = [self.energy_demand] * N
        sessions_dict = session_generator(
            num_sessions=N,
            arrivals=[0] * N,
            departures=[self.horizon] * N,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * N,
        )
        infra_dict = three_phase_balanced_network(
            evses_per_phase=N // 3, limit=32 * N / 3
        )
        interface = self.interface(sessions_dict, infra_dict)

        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()

        obj = [ObjectiveComponent(quick_charge), ObjectiveComponent(equal_share, 1e-12)]

        asa = AdaptiveChargingOptimization(obj, interface, constraint_type="SOC")

        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infrastructure)
        print(time.time() - start_time)


class TestLargeFeasibleThreePhaseLinear(BaseAlgoTestScenarios):
    def setUp(self):
        super().setUp()
        self.energy_demand = 10
        self.horizon = 12 * 12
        N = 54
        energy_demand = [self.energy_demand] * N
        sessions_dict = session_generator(
            num_sessions=N,
            arrivals=[0] * N,
            departures=[self.horizon] * N,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * N,
        )
        infra_dict = three_phase_balanced_network(
            evses_per_phase=N // 3, limit=32 * N / 3
        )
        interface = self.interface(sessions_dict, infra_dict)

        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()

        asa = AdaptiveChargingOptimization(
            DEFAULT_OBJECTIVE, interface, constraint_type="LINEAR"
        )
        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infrastructure)
        print(time.time() - start_time)


class TestTOUCostMinimizationTinyNetwork(BaseAlgoTestScenarios):
    def interface(self, sessions_dict, infra_dict):
        iface = TestingInterface(
            {
                "active_sessions": sessions_dict,
                "infrastructure_info": infra_dict,
                "current_time": self.current_time,
                "period": self.period,
            }
        )
        iface.get_prices = Mock(return_value=np.array([0.3] * 6 + [0.1] * 6))
        return iface

    def build_and_run(self, session_dict, infra_dict, energy_equality=False):
        interface = self.interface(session_dict, infra_dict)
        self.infrastructure = interface.infrastructure_info()
        self.sessions = interface.active_sessions()
        objective = [ObjectiveComponent(tou_energy_cost)]
        asa = AdaptiveChargingOptimization(
            objective, interface, enforce_energy_equality=energy_equality
        )
        self.rates = asa.solve(self.sessions, self.infrastructure)

    def setUp(self):
        super().setUp()
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[self.horizon] * 2,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        self.build_and_run(sessions_dict, infra_dict, True)

    def test_dont_charge_at_high_cost(self):
        assert np.allclose(self.rates[:, :6], 0, atol=1e-3)


class TestTOUCostMinimizationTinyNetworkNonZeroCurrentTime(
    TestTOUCostMinimizationTinyNetwork
):
    def interface(self, sessions_dict, infra_dict):
        iface = TestingInterface(
            {
                "active_sessions": sessions_dict,
                "infrastructure_info": infra_dict,
                "current_time": self.current_time,
                "period": self.period,
            }
        )
        iface.get_prices = Mock(return_value=np.array([0.3] * 2 + [0.1] * 6))
        return iface

    def setUp(self):
        self.period = 5  # minutes
        self.max_rate = 32  # A
        self.energy_demand = 3.3  # kW
        self.horizon = 12
        self.current_time = 4
        energy_demand = [self.energy_demand] * 2
        sessions_dict = session_generator(
            num_sessions=2,
            arrivals=[0] * 2,
            departures=[self.horizon] * 2,
            remaining_energy=energy_demand,
            requested_energy=energy_demand,
            max_rates=[self.max_rate] * 2,
        )
        infra_dict = single_phase_single_constraint(num_evses=2, limit=64)
        self.build_and_run(sessions_dict, infra_dict, True)

    def test_dont_charge_at_high_cost(self):
        assert np.allclose(self.rates[:, :2], 0, atol=1e-3)
        assert np.all(self.rates[:, 2:] > 1e-4)


# Delete BaseAlgoTestScenarios since it should never be run on its own.
del BaseAlgoTestScenarios
