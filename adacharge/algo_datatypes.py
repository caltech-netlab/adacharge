class SessionInfo:
    """ Simple class to store information relevant to a charging session.

    Args:
        station_id:
        session_id:
        remaining_energy:
        remaining_time:
        arrival_offset:
        min_rates:
        max_rates:
    """
    def __init__(self, station_id, session_id, energy_requested, energy_delivered, arrival, departure, current_time=0,
                 min_rates=0, max_rates=float('inf')):
        self.station_id = station_id
        self.session_id = session_id
        self.energy_requested = energy_requested
        self.energy_delivered = energy_delivered
        self.arrival = arrival
        self.departure = departure
        self.current_time = current_time
        self.min_rates = min_rates
        self.max_rates = max_rates

    @property
    def remaining_energy(self):
        return self.energy_requested - self.energy_delivered

    @property
    def remaining_time(self):
        return max(self.departure - self.current_time, 0)

    @property
    def arrival_offset(self):
        return max(self.arrival - self.current_time, 0)


class InfrastructureInfo:
    def __init__(self, constraint_matrix, magnitudes, phases, constraint_index, eves_index,
                 max_pilot, min_pilot, allowable_pilots=None):
        self.constraint_matrix = constraint_matrix
        self.magnitudes = magnitudes
        self.phases = phases
        self.constraint_index = constraint_index
        self.evse_index = eves_index
        self.max_pilot = max_pilot
        self.min_pilot = min_pilot
        self.allowable_pilots = allowable_pilots
