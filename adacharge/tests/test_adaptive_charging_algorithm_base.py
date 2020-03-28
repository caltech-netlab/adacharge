from unittest import TestCase
from adaptive_charging_algorithm_base import *
from algo_datatypes import *
import time
from testing_utilities import *

DEFAULT_OBJECTIVE = [ObjectiveComponent(quick_charge)]


class BaseAlgoTestScenarios(TestCase):
    def setUp(self) -> None:
        self.max_rate = 1
        self.energy_demand = 5
        self.sessions = []
        self.infra = None
        self.rates = None

    def test_all_rates_less_than_limit(self):
        assert (self.rates <= self.max_rate + 1e-7).all()

    def test_all_energy_demands_met(self):
        energy_expected = np.zeros(self.rates.shape[0])
        energy_delivered = np.zeros(self.rates.shape[0])
        for s in self.sessions:
            i = self.infra.evse_index.index(s.station_id)
            energy_expected[i] = s.remaining_energy
            energy_delivered[i] = self.rates[i, s.arrival_offset: s.arrival_offset + s.remaining_time].sum()
        assert np.allclose(energy_delivered, energy_expected)

    def test_no_charging_when_not_plugged_in(self):
        not_plugged_in = np.ones(self.rates.shape, dtype=bool)
        for s in self.sessions:
            i = self.infra.evse_index.index(s.station_id)
            not_plugged_in[i, s.arrival_offset: s.arrival_offset + s.remaining_time] = 0
        assert np.allclose(self.rates[not_plugged_in], 0)

    def test_infrastructure_constraints_satisfied(self):
        phase_in_rad = np.deg2rad(self.infra.phases)
        for j, v in enumerate(self.infra.constraint_matrix):
            a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
            line_currents = np.linalg.norm(a @ self.rates, axis=0)
            assert np.all(line_currents <= self.infra.magnitudes[j] + 1e-7)


# Simple Correctness Tests
class TestTinyFeasibleNetwork(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        self.max_rate = 1
        self.energy_demand = 5
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE)
        self.sessions = session_generator(2, [0]*2, [10]*2, [self.energy_demand]*2, max_rates=[self.max_rate]*2)
        self.infra = single_phase_network(N=2, limit=2)
        self.rates = asa.solve(self.sessions, self.infra)


class TestTinyFeasibleNetworkEnergyEquality(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        self.max_rate = 1
        self.energy_demand = 5
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE, enforce_energy_equality=True)
        self.sessions = session_generator(2, [0]*2, [10]*2, [self.energy_demand]*2, max_rates=[self.max_rate]*2)
        self.infra = single_phase_network(N=2, limit=2)
        self.rates = asa.solve(self.sessions, self.infra)


class TestTinyInfeasibleNetworkEnergyEquality(TestCase):
    def test_infeasible_input_with_equality_constraints(self):
        asa = AdaptiveChargingAlgorithmBase([ObjectiveComponent(quick_charge)], enforce_energy_equality=True)
        self.sessions = session_generator(2, [0]*2, [10, 4], [5]*2, max_rates=[1]*2)
        self.infra = single_phase_network(N=2, limit=2)
        with self.assertRaises(InfeasibilityException):
            _ = asa.solve(self.sessions, self.infra)


class TestTinyFeasibleNetworkDelayedStart(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        self.max_rate = 1
        self.energy_demand = 5
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE,)
        self.sessions = session_generator(2, [0, 4], [10, 14], [self.energy_demand]*2, max_rates=[self.max_rate]*2)
        self.infra = single_phase_network(N=2, limit=2)
        self.rates = asa.solve(self.sessions, self.infra)


class TestTinyFeasibleMultipleSessionsSameEVSE(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        self.max_rate = 1
        self.energy_demand = 5
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE,)
        self.sessions = session_generator(2, [0, 12], [10, 22], [self.energy_demand]*2, station_ids=['0']*2,
                                          max_rates=[self.max_rate]*2)
        self.infra = single_phase_network(N=2, limit=2)
        self.rates = asa.solve(self.sessions, self.infra)


class TestTinyMinimumCharge(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        self.max_rate = 1
        self.min_rate = 0.5
        self.energy_demand = 5
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE,)
        self.sessions = session_generator(2, [0]*2, [10]*2, [self.energy_demand]*2, min_rates=[self.min_rate]*2,
                                          max_rates=[self.max_rate]*2)
        self.infra = single_phase_network(N=2, limit=2)
        self.rates = asa.solve(self.sessions, self.infra)

    def test_all_rates_less_than_limit(self):
        assert (self.rates >= self.min_rate - 1e-7).all()


class TestTinyPeakLimitScalar(BaseAlgoTestScenarios):
    def setUp(self) -> None:
        self.max_rate = 1
        self.energy_demand = 5
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE,)
        self.sessions = session_generator(2, [0]*2, [10]*2, [self.energy_demand]*2,
                                          max_rates=[self.max_rate]*2)
        self.infra = single_phase_network(N=2, limit=2)
        self.peak_limit = 1
        self.rates = asa.solve(self.sessions, self.infra, self.peak_limit)

    def test_peak_less_than_limit(self):
        assert (self.rates.sum(axis=0) <= self.peak_limit + 1e-7).all()


class TestTinyPeakLimitVector(TestTinyPeakLimitScalar):
    def setUp(self) -> None:
        self.max_rate = 1
        self.energy_demand = 5
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE,)
        self.sessions = session_generator(2, [0]*2, [10]*2, [self.energy_demand]*2,
                                          max_rates=[self.max_rate]*2)
        self.infra = single_phase_network(N=2, limit=2)
        self.peak_limit = np.array([1.5]*5 + [1]*5)
        self.rates = asa.solve(self.sessions, self.infra, self.peak_limit)


# Basic Stress Tests
class TestLargeFeasibleSinglePhase(BaseAlgoTestScenarios):
    def setUp(self):
        self.max_rate = 32
        self.energy_demand = 32 * 12 * 3
        N = 54
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE, constraint_type='LINEAR')
        self.sessions = session_generator(N, [0]*N, [144]*N, [self.energy_demand]*N, max_rates=[self.max_rate]*N)
        self.infra = single_phase_network(N, 32*N/3)
        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infra)
        print(time.time() - start_time)


class TestLargeFeasibleSinglePhaseOSQP(BaseAlgoTestScenarios):
    def setUp(self):
        self.max_rate = 32
        self.energy_demand = 32 * 12 * 3
        N = 54
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE, constraint_type='LINEAR', solver=cp.OSQP)
        self.sessions = session_generator(N, [0]*N, [144]*N, [self.energy_demand]*N, max_rates=[self.max_rate]*N)
        self.infra = single_phase_network(N, 32*N/3)
        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infra)
        print(time.time() - start_time)


class TestLargeFeasibleThreePhaseSOC(BaseAlgoTestScenarios):
    def setUp(self):
        self.max_rate = 32
        self.energy_demand = 32 * 12 * 3
        N = 54
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE)
        self.sessions = session_generator(N, [0]*N, [144]*N, [self.energy_demand]*N, max_rates=[self.max_rate]*N)
        self.infra = three_phase_balanced_network(N // 3, 32 * N / 3)
        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infra)
        print(time.time() - start_time)


class TestLargeFeasibleThreePhaseLinear(BaseAlgoTestScenarios):
    def setUp(self):
        self.max_rate = 32
        self.energy_demand = 32 * 12 * 3
        N = 54
        asa = AdaptiveChargingAlgorithmBase(DEFAULT_OBJECTIVE, constraint_type='LINEAR')
        self.sessions = session_generator(N, [0]*N, [144]*N, [self.energy_demand]*N, max_rates=[self.max_rate]*N)
        self.infra = three_phase_balanced_network(N // 3, 32 * N / 3)
        start_time = time.time()
        self.rates = asa.solve(self.sessions, self.infra)
        print(time.time() - start_time)


# Delete BaseAlgoTestScenarios since it should never be run on its own.
del BaseAlgoTestScenarios
