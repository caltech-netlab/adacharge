import numpy as np
from unittest import TestCase
from unittest.mock import Mock
from adacharge.postprocessing import *

# from testing_utilities import *
from acnportal.algorithms.tests.generate_test_cases import (
    session_generator,
    single_phase_single_constraint,
    three_phase_balanced_network,
)
from acnportal.algorithms.tests.testing_interface import TestingInterface
from numpy import testing as nptest
from acnportal.algorithms import earliest_deadline_first


class TestFloorToSet(TestCase):
    def test_x_in_set(self):
        x = 5
        x_prime = floor_to_set(x, np.array([0, 5, 10]))
        self.assertEqual(x_prime, x)

    def test_x_in_set_eps_0(self):
        x = 5
        x_prime = floor_to_set(x, np.array([0, 5, 10]), eps=0)
        self.assertEqual(x_prime, x)

    def test_round_down(self):
        x = 4.9
        x_prime = floor_to_set(x, np.array([0, 5, 10]), eps=0.05)
        self.assertEqual(x_prime, 0)

    def test_round_up_within_eps(self):
        x = 4.98
        x_prime = floor_to_set(x, np.array([0, 5, 10]), eps=0.05)
        self.assertEqual(x_prime, 5)

    def test_less_than_minimum_allowable(self):
        x = -1
        x_prime = floor_to_set(x, np.array([0, 5, 10]), eps=0.05)
        self.assertEqual(x_prime, 0)

    def test_greater_than_max_allowable(self):
        x = 15
        x_prime = floor_to_set(x, np.array([0, 5, 10]), eps=0.05)
        self.assertEqual(x_prime, 10)


class TestCeilToSet(TestCase):
    def test_x_in_set(self):
        x = 5
        x_prime = ceil_to_set(x, np.array([0, 5, 10]))
        self.assertEqual(x_prime, x)

    def test_x_in_set_eps_0(self):
        x = 5
        x_prime = ceil_to_set(x, np.array([0, 5, 10]), eps=0)
        self.assertEqual(x_prime, x)

    def test_round_up(self):
        x = 2.5
        x_prime = ceil_to_set(x, np.array([0, 5, 10]), eps=0.05)
        self.assertEqual(x_prime, 5)

    def test_round_down_within_eps(self):
        x = 5.02
        x_prime = ceil_to_set(x, np.array([0, 5, 10]), eps=0.05)
        self.assertEqual(x_prime, 5)

    def test_less_than_minimum_allowable(self):
        x = -1
        x_prime = ceil_to_set(x, np.array([0, 5, 10]), eps=0.05)
        self.assertEqual(x_prime, 0)

    def test_greater_than_max_allowable(self):
        x = 15
        x_prime = ceil_to_set(x, np.array([0, 5, 10]), eps=0.05)
        self.assertEqual(x_prime, 10)


class TestIncrementInSet(TestCase):
    def test_x_in_set(self):
        x = 5
        x_prime = increment_in_set(x, np.array([0, 5, 10]))
        self.assertEqual(x_prime, 10)

    def test_round_up(self):
        x = 2.5
        x_prime = increment_in_set(x, np.array([0, 5, 10]))
        self.assertEqual(x_prime, 5)

    def test_less_than_minimum_allowable(self):
        x = -1
        x_prime = increment_in_set(x, np.array([0, 5, 10]))
        self.assertEqual(x_prime, 0)

    def test_greater_than_max_allowable(self):
        x = 15
        x_prime = increment_in_set(x, np.array([0, 5, 10]))
        self.assertEqual(x_prime, 10)


class TestProjectIntoContinuousFeasiblePilots(TestCase):
    def setUp(self):
        self.infrastructure = Mock()
        self.infrastructure.max_pilot = np.full(5, 32)
        self.infrastructure.min_pilot = np.full(5, 0)
        self.infrastructure.num_stations = 5

    def test_rates_are_feasible(self):
        rates = np.full((5, 20), 16)
        processed = project_into_continuous_feasible_pilots(rates, self.infrastructure)
        nptest.assert_equal(processed, 16)

    def test_rates_greater_than_limit(self):
        rates = np.full((5, 20), 33)
        processed = project_into_continuous_feasible_pilots(rates, self.infrastructure)
        nptest.assert_equal(processed, 32)

    def test_rates_less_than_limit(self):
        rates = np.full((5, 20), -1)
        processed = project_into_continuous_feasible_pilots(rates, self.infrastructure)
        nptest.assert_equal(processed, 0)


class TestProjectIntoDiscreteFeasiblePilots(TestCase):
    def setUp(self):
        self.infrastructure = Mock()
        self.infrastructure.max_pilot = np.full(5, 32)
        self.infrastructure.min_pilot = np.full(5, 0)
        self.infrastructure.allowable_pilots = [[0, 8, 16, 24, 32]] * 5
        self.infrastructure.num_stations = 5

    def test_rates_are_feasible(self):
        rates = np.full((5, 20), 16)
        processed = project_into_discrete_feasible_pilots(rates, self.infrastructure)
        nptest.assert_equal(processed, 16)

    def test_rates_within_range_but_not_allowable(self):
        rates = np.full((5, 20), 18)
        processed = project_into_discrete_feasible_pilots(rates, self.infrastructure)
        nptest.assert_equal(processed, 16)

    def test_rates_within_range_but_not_allowable_round_up(self):
        rates = np.full((5, 20), 15.98)
        processed = project_into_discrete_feasible_pilots(rates, self.infrastructure)
        nptest.assert_equal(processed, 16)

    def test_rates_greater_than_limit(self):
        rates = np.full((5, 20), 33)
        processed = project_into_discrete_feasible_pilots(rates, self.infrastructure)
        nptest.assert_equal(processed, 32)

    def test_rates_less_than_limit(self):
        rates = np.full((5, 20), -1)
        processed = project_into_discrete_feasible_pilots(rates, self.infrastructure)
        nptest.assert_equal(processed, 0)


class TestIndexBasedReallocation(TestCase):
    def setUp(self):
        self.horizon = 10
        self.sessions = session_generator(
            num_sessions=3,
            arrivals=[0] * 3,
            departures=[2, 3, 4],
            remaining_energy=[3.3] * 3,
            requested_energy=[3.3] * 3,
            max_rates=[32] * 3,
            min_rates=[0] * 3,
        )

    def test_no_reallocation_peak_binding(self):
        infrastructure = single_phase_single_constraint(
            num_evses=3,
            limit=66,
            allowable_pilots=[np.array([0, 8, 16, 24, 32]) for _ in range(3)],
        )
        interface = TestingInterface(
            {
                "active_sessions": self.sessions,
                "infrastructure_info": infrastructure,
                "current_time": 0,
                "period": 5,
            }
        )
        rates = np.full((3, self.horizon), 16)
        peak = 16 * 3
        processed = index_based_reallocation(
            rates,
            interface.active_sessions(),
            interface.infrastructure_info(),
            peak,
            earliest_deadline_first,
            interface,
        )
        nptest.assert_equal(processed, 16)

    def test_reallocate_to_peak_infrastructure_not_binding(self):
        infrastructure = single_phase_single_constraint(
            num_evses=3,
            limit=66,
            allowable_pilots=[np.array([0] + list(range(8, 33))) for _ in range(3)],
        )
        interface = TestingInterface(
            {
                "active_sessions": self.sessions,
                "infrastructure_info": infrastructure,
                "current_time": 0,
                "period": 5,
            }
        )
        rates = np.full((3, self.horizon), 16)
        peak = 16 * 3 + 2
        processed = index_based_reallocation(
            rates,
            interface.active_sessions(),
            interface.infrastructure_info(),
            peak,
            earliest_deadline_first,
            interface,
        )
        expected = np.full((3, self.horizon), 16)
        expected[:2, 0] = 17
        nptest.assert_equal(processed, expected)

    def test_reallocate_infrastructure_binding_single_phase(self):
        infrastructure = single_phase_single_constraint(
            num_evses=3,
            limit=49,
            allowable_pilots=[np.array([0] + list(range(8, 33))) for _ in range(3)],
        )
        interface = TestingInterface(
            {
                "active_sessions": self.sessions,
                "infrastructure_info": infrastructure,
                "current_time": 0,
                "period": 5,
            }
        )
        rates = np.full((3, self.horizon), 16)
        peak = 60
        processed = index_based_reallocation(
            rates,
            interface.active_sessions(),
            interface.infrastructure_info(),
            peak,
            earliest_deadline_first,
            interface,
        )
        expected = np.full((3, self.horizon), 16)
        expected[0, 0] = 17
        nptest.assert_equal(processed, expected)

    def test_reallocate_infrastructure_binding_three_phase(self):
        infrastructure = three_phase_balanced_network(
            evses_per_phase=1,
            limit=16.51 * np.sqrt(3),
            allowable_pilots=[np.array([0] + list(range(8, 33))) for _ in range(3)],
        )
        interface = TestingInterface(
            {
                "active_sessions": self.sessions,
                "infrastructure_info": infrastructure,
                "current_time": 0,
                "period": 5,
            }
        )
        rates = np.full((3, self.horizon), 16)
        peak = 60
        processed = index_based_reallocation(
            rates,
            interface.active_sessions(),
            interface.infrastructure_info(),
            peak,
            earliest_deadline_first,
            interface,
        )
        expected = np.full((3, self.horizon), 16)
        expected[0, 0] = 17
        nptest.assert_equal(processed, expected)

    def test_reallocate_to_peak_energy_binding(self):
        self.sessions = session_generator(
            num_sessions=3,
            arrivals=[0] * 3,
            departures=[2, 3, 4],
            remaining_energy=[0.277, 3.3, 3.3],
            requested_energy=[3.3] * 3,
            max_rates=[32] * 3,
            min_rates=[0] * 3,
        )
        infrastructure = single_phase_single_constraint(
            num_evses=3,
            limit=66,
            allowable_pilots=[np.array([0] + list(range(8, 33))) for _ in range(3)],
        )
        interface = TestingInterface(
            {
                "active_sessions": self.sessions,
                "infrastructure_info": infrastructure,
                "current_time": 0,
                "period": 5,
            }
        )
        rates = np.full((3, self.horizon), 16)
        peak = 16 * 3 + 2
        processed = index_based_reallocation(
            rates,
            interface.active_sessions(),
            interface.infrastructure_info(),
            peak,
            earliest_deadline_first,
            interface,
        )
        expected = np.full((3, self.horizon), 16)
        expected[1:, 0] = 17
        nptest.assert_equal(processed, expected)
