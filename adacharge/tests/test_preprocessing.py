from unittest import TestCase
from unittest.mock import Mock
from preprocessing import *
from test_utils import *

ARRIVAL_TIME = 0
SESSION_DUR = 5
ENERGY_DEMAND = 32*5
N = 3


class TestApplyRampdown(TestCase):
    def test_default_max_rates_scalars(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME]*N, [SESSION_DUR]*N, [ENERGY_DEMAND]*N, [32]*N)
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 16 for i in range(N)})
        modified_sessions = apply_rampdown(rd, active_sessions)
        for session in modified_sessions:
            assert np.allclose(session.max_rates, 16)
            assert np.allclose(session.min_rates, 0)

    def test_lower_existing_max_scalars(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME]*N, [SESSION_DUR]*N, [ENERGY_DEMAND]*N, [12]*N)
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 16 for i in range(N)})
        modified_sessions = apply_rampdown(rd, active_sessions)
        for session in modified_sessions:
            assert np.allclose(session.max_rates, 12)
            assert np.allclose(session.min_rates, 0)

    def test_vector_existing_max_scalar_rampdown(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME]*N, [SESSION_DUR]*N, [ENERGY_DEMAND]*N, [32]*N)
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 16 for i in range(N)})
        modified_sessions = apply_rampdown(rd, active_sessions)
        for session in modified_sessions:
            assert np.allclose(session.max_rates, 16)
            assert np.allclose(session.min_rates, 0)

    def test_vector_lower_existing_max_scalar_rampdown(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME]*N, [SESSION_DUR]*N, [ENERGY_DEMAND]*N, [[12]*5]*N)
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 16 for i in range(N)})
        modified_sessions = apply_rampdown(rd, active_sessions)
        for session in modified_sessions:
            assert np.allclose(session.max_rates, 12)
            assert np.allclose(session.min_rates, 0)

    def test_all_vectors_rampdown_lower(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME]*N, [SESSION_DUR]*N, [ENERGY_DEMAND]*N, [[32]*5]*N)
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': [16]*5 for i in range(N)})
        modified_sessions = apply_rampdown(rd, active_sessions)
        for session in modified_sessions:
            assert np.allclose(session.max_rates, 16)
            assert np.allclose(session.min_rates, 0)

    def test_all_vectors_existing_lower(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME]*N, [SESSION_DUR]*N, [ENERGY_DEMAND]*N, [[12]*5]*N)
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': [16]*5 for i in range(N)})
        modified_sessions = apply_rampdown(rd, active_sessions)
        for session in modified_sessions:
            assert np.allclose(session.max_rates, 12)
            assert np.allclose(session.min_rates, 0)

    def test_minimum_rates_binding(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME]*N, [SESSION_DUR]*N, [ENERGY_DEMAND]*N, [12]*N, [8]*N)
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 6 for i in range(N)})
        modified_sessions = apply_rampdown(rd, active_sessions)
        for session in modified_sessions:
            assert np.allclose(session.max_rates, 8)
            assert np.allclose(session.min_rates, 8)


class TestApplyMinimumChargingRate(TestCase):
    def test_apply_min_evse_less_than_session_min(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME] * N, [SESSION_DUR] * N, [ENERGY_DEMAND] * N, [32] * N)
        infrastructure = single_phase_network(N, 32, 32, 8)
        modified_sessions = apply_minimum_charging_rate(active_sessions, infrastructure)
        for session in modified_sessions:
            assert np.allclose(session.min_rates[0], 8)
            assert np.allclose(session.min_rates[1:], 0)

    def test_apply_min_evse_greater_than_session_max(self):
        active_sessions = session_generator(N, [ARRIVAL_TIME] * N, [SESSION_DUR] * N, [ENERGY_DEMAND] * N, [6] * N)
        infrastructure = single_phase_network(N, 32, 32, 8)
        modified_sessions = apply_minimum_charging_rate(active_sessions, infrastructure)
        for session in modified_sessions:
            assert np.allclose(session.min_rates[0], 8)
            assert np.allclose(session.min_rates[1:], 0)
            assert np.allclose(session.max_rates[0], 8)
            assert np.allclose(session.max_rates[1:], 6)

    def test_apply_min_infeasible(self):
        _N = 3
        active_sessions = session_generator(_N, [0, 1, 2], [SESSION_DUR] * _N, [ENERGY_DEMAND] * _N, [32] * _N)
        infrastructure = single_phase_network(_N, 16, 32, 8)
        modified_sessions = apply_minimum_charging_rate(active_sessions, infrastructure)
        for i in range(2):
            assert np.allclose(modified_sessions[i].min_rates[0], 8)
            assert np.allclose(modified_sessions[i].min_rates[1:], 0)
        # It is not feasible to deliver 8 A to session '2', so max and min should be 0 at time t=0.
        assert np.allclose(modified_sessions[2].min_rates, 0)
        assert np.allclose(modified_sessions[2].max_rates[0], 0)