import numpy as np


class SessionInfo:
    """ Simple class to store information relevant to a charging session.

        Args:
            station_id:
            session_id:
            energy_requested:
            energy_delivered:
            arrival:
            departure:
            current_time:
            min_rates:
            max_rates:
    """
    def __init__(self, station_id, session_id, energy_requested, energy_delivered, arrival, departure,
                 current_time=0, min_rates=0, max_rates=float('inf')):
        self.station_id = station_id
        self.session_id = session_id
        self.energy_requested = energy_requested
        self.energy_delivered = energy_delivered
        self.arrival = arrival
        self.departure = departure
        self.current_time = current_time
        self.min_rates = np.array([min_rates] * self.remaining_time) if np.isscalar(min_rates) else min_rates
        self.max_rates = np.array([max_rates] * self.remaining_time) if np.isscalar(max_rates) else max_rates

    @property
    def remaining_energy(self):
        return self.energy_requested - self.energy_delivered

    @property
    def arrival_offset(self):
        return max(self.arrival - self.current_time, 0)

    @property
    def remaining_time(self):
        offset = max(self.arrival_offset, self.current_time)
        return max(self.departure - offset, 0)


class InfrastructureInfo:
    def __init__(self, constraint_matrix, constraint_limits, phases, voltages, constraint_index, eves_index,
                 max_pilot, min_pilot, allowable_pilots=None):
        self.constraint_matrix = constraint_matrix
        self.constraint_limits = constraint_limits
        self.phases = phases
        self.voltages = voltages
        self.constraint_index = constraint_index
        self.evse_index = eves_index
        self._evse_index_dict = {station_id: i for i, station_id in enumerate(self.evse_index)}
        self.max_pilot = max_pilot
        self.min_pilot = min_pilot
        self.allowable_pilots = allowable_pilots if allowable_pilots is not None else [None] * self.num_stations

    @property
    def num_stations(self):
        return len(self.evse_index)

    def get_station_index(self, station_id):
        return self._evse_index_dict[station_id]
