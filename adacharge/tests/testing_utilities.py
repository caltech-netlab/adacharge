import numpy as np
from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo


# def session_generator(N, arrivals, departures, remaining_energy, max_rates, min_rates=None, station_ids=None,
#                       current_time=0):
#     sessions = []
#     for i in range(N):
#         station_id = station_ids[i] if station_ids is not None else f'{i}'
#         session_id = f'{i}'
#         min_rate = min_rates[i] if min_rates is not None else 0
#         s = SessionInfo(station_id, session_id, remaining_energy[i], 0, arrivals[i], departures[i], current_time,
#                         min_rate, max_rates[i])
#         sessions.append(s)
#     return sessions


def session_generator(num_sessions, arrivals, departures, remaining_energy, max_rates,
                      requested_energy=None, min_rates=None,
                      station_ids=None, current_time=0, as_dict=False):
    sessions = []
    for i in range(num_sessions):
        station_id = station_ids[i] if station_ids is not None else f'{i}'
        session_id = f'{i}'
        min_rate = min_rates[i] if min_rates is not None else 0
        if requested_energy is None:
            requested_energy = remaining_energy
        s = {'station_id': station_id,
             'session_id': session_id,
             'requested_energy': requested_energy[i],
             'energy_delivered': requested_energy[i] - remaining_energy[i],
             'arrival': arrivals[i],
             'departure': departures[i],
             'estimated_departure': None,
             'min_rates': min_rate,
             'max_rates': max_rates[i],
             'current_time': current_time}
        sessions.append(s)
    if as_dict:
        return sessions
    else:
        return [SessionInfo(**s) for s in sessions]



def single_phase_network(N, limit, max_pilot=32, min_pilot=8):
    return InfrastructureInfo(np.array([[1]*N]),
                              np.array([limit]),
                              np.array([0]*N),
                              np.array([208]*N),
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
                              np.array([208] * (3 * N)),
                              ['AB', 'BC', 'CA'],
                              [f'{i}' for i in range(3 * N)],
                              {f'{i}': max_pilot for i in range(3 * N)},
                              {f'{i}': min_pilot for i in range(3 * N)})


