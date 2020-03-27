import numpy as np
from algo_datatypes import SessionInfo, InfrastructureInfo


def session_generator(N, arrivals, departures, remaining_energy, max_rates, min_rates=None, station_ids=None):
    sessions = []
    for i in range(N):
        station_id = station_ids[i] if station_ids is not None else f'{i}'
        session_id = f'{i}'
        min_rate = min_rates[i] if min_rates is not None else 0
        s = SessionInfo(station_id, session_id, remaining_energy[i], 0, arrivals[i], departures[i], 0,
                        min_rate, max_rates[i])
        sessions.append(s)
    return sessions


def single_phase_network(N, limit, max_pilot=32, min_pilot=8):
    return InfrastructureInfo(np.array([[1]*N]),
                              np.array([limit]),
                              np.array([0]*N),
                              ['all'],
                              [f'{i}' for i in range(N)],
                              {f'{i}': max_pilot for i in range(N)},
                              {f'{i}': min_pilot for i in range(N)})


def three_phase_balanced_network(N, limit, max_pilot=32, min_pilot=8):
    return InfrastructureInfo(np.array([[1]  * N + [-1] * N + [0]  * N,
                                        [0]  * N + [1]  * N + [-1] * N,
                                        [-1] * N + [0]  * N + [1]  * N]),
                              np.array([limit]*3),
                              np.array([0] * N + [-120] * N + [120] * N),
                              ['AB', 'BC', 'CA'],
                              [f'{i}' for i in range(3 * N)],
                              {f'{i}': max_pilot for i in range(3 * N)},
                              {f'{i}': min_pilot for i in range(3 * N)})


