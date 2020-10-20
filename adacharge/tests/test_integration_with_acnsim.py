from unittest import TestCase
from unittest.mock import Mock

import pytz
from datetime import datetime
from acnportal import acnsim
from adacharge import *


class AdaptiveSchedulingAlgorithmBase(TestCase):
    # These class variables should be set in the setUpClass() method
    alg = None
    sim = None

    @classmethod
    def test_infrastructure_constraints_satisfied(cls):
        assert cls.sim.network.is_feasible(cls.sim.pilot_signals)
        assert cls.sim.network.is_feasible(cls.sim.charging_rates)

    @classmethod
    def test_all_rates_less_than_limit(cls):
        eps = 1e-3
        station_ids = cls.sim.network.station_ids
        for ev in cls.sim.ev_history.values():
            i = station_ids.index(ev.station_id)
            pilot_power = (
                cls.sim.pilot_signals[i, ev.arrival : ev.departure]
                * cls.sim.network.voltages[ev.station_id]
                / 1000
                * (5 / 60)
            )
            assert np.all(
                pilot_power <= cls.sim.network._EVSEs[ev.station_id].max_rate + eps
            )
            assert np.all(pilot_power <= ev.maximum_charging_power + eps)

    @classmethod
    def test_all_energy_demands_met(cls):
        assert acnsim.analysis.proportion_of_energy_delivered(cls.sim) >= 0.9999

    @classmethod
    def test_no_charging_when_not_plugged_in(cls):
        not_plugged_in = np.ones(cls.sim.pilot_signals.shape, dtype=bool)
        station_ids = cls.sim.network.station_ids
        for ev in cls.sim.ev_history.values():
            i = station_ids.index(ev.station_id)
            not_plugged_in[i, ev.arrival : ev.departure] = 0
        assert np.allclose(cls.sim.pilot_signals[not_plugged_in], 0)


class TestACASingleEV(AdaptiveSchedulingAlgorithmBase):
    @classmethod
    def setUpClass(cls) -> None:
        timezone = pytz.timezone("America/Los_Angeles")
        start = timezone.localize(datetime(2018, 9, 1))
        period = 5  # minute
        voltage = 208  # volts

        cn = acnsim.ChargingNetwork()
        cn.register_evse(acnsim.get_evse_by_type("PS-1", "BASIC"), voltage, 0)
        cn.add_constraint(acnsim.Current("PS-1"), 100)

        batt = acnsim.Battery(100, 0, 7)
        ev = acnsim.EV(5, 5 + 12, 6.6, "PS-1", "test", batt)
        events = acnsim.EventQueue([acnsim.PluginEvent(ev.arrival, ev)])

        quick_charge_obj = [
            ObjectiveComponent(quick_charge),
            ObjectiveComponent(equal_share, 1e-12),
        ]
        cls.alg = AdaptiveSchedulingAlgorithm(quick_charge_obj, solver=cp.ECOS)

        cls.sim = acnsim.Simulator(
            cn, cls.alg, events, start, period=period, verbose=False
        )
        cls.sim.run()


class TestACACaltechOneDay(AdaptiveSchedulingAlgorithmBase):
    @classmethod
    def setUpClass(cls) -> None:
        API_KEY = "DEMO_TOKEN"
        tz = pytz.timezone("America/Los_Angeles")
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = "caltech"
        basic_evse = True
        start = "9-01-2018"
        end = "9-02-2018"
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, "%m-%d-%Y"))
        end_time = tz.localize(datetime.strptime(end, "%m-%d-%Y"))

        events = acnsim.acndata_events.generate_events(
            API_KEY,
            site,
            start_time,
            end_time,
            period,
            voltage,
            default_battery_power,
            force_feasible=force_feasible,
        )

        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)

        quick_charge_obj = [
            ObjectiveComponent(quick_charge),
            ObjectiveComponent(equal_share, 1e-12),
        ]
        cls.alg = AdaptiveSchedulingAlgorithm(quick_charge_obj, solver=cp.ECOS)

        cls.sim = acnsim.Simulator(
            cn, cls.alg, events, start_time, period=period, verbose=False
        )
        cls.sim.run()


class TestACACaltechSingleDayQuantized(AdaptiveSchedulingAlgorithmBase):
    @classmethod
    def setUpClass(cls) -> None:
        API_KEY = "DEMO_TOKEN"
        tz = pytz.timezone("America/Los_Angeles")
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = "caltech"
        basic_evse = False
        start = "9-01-2018"
        end = "9-02-2018"
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, "%m-%d-%Y"))
        end_time = tz.localize(datetime.strptime(end, "%m-%d-%Y"))

        events = acnsim.acndata_events.generate_events(
            API_KEY,
            site,
            start_time,
            end_time,
            period,
            voltage,
            default_battery_power,
            force_feasible=force_feasible,
        )
        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)
        quick_charge_obj = [
            ObjectiveComponent(quick_charge),
            ObjectiveComponent(equal_share, 1e-12),
        ]

        cls.alg = AdaptiveSchedulingAlgorithm(
            quick_charge_obj, solver=cp.ECOS, quantize=True
        )
        cls.sim = acnsim.Simulator(
            cn, cls.alg, events, start_time, period=period, verbose=False
        )
        cls.sim.run()

    @classmethod
    def test_all_energy_demands_met(cls):
        # Because of quantization we need to relax the energy delivered requirement
        assert acnsim.analysis.proportion_of_energy_delivered(cls.sim) >= 0.99

    @classmethod
    def test_all_rates_in_allowable(cls):
        eps = 1e-3
        station_ids = cls.sim.network.station_ids
        for ev in cls.sim.ev_history.values():
            i = station_ids.index(ev.station_id)
            pilot = cls.sim.pilot_signals[i, ev.arrival : ev.departure]
            assert np.all(
                np.isin(
                    pilot, cls.sim.network._EVSEs[ev.station_id].allowable_pilot_signals
                )
            )


class TestACACaltechSingleDayQuantizedReallocated(TestACACaltechSingleDayQuantized):
    @classmethod
    def setUpClass(cls) -> None:
        API_KEY = "DEMO_TOKEN"
        tz = pytz.timezone("America/Los_Angeles")
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = "caltech"
        basic_evse = False
        start = "9-01-2018"
        end = "9-02-2018"
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, "%m-%d-%Y"))
        end_time = tz.localize(datetime.strptime(end, "%m-%d-%Y"))

        events = acnsim.acndata_events.generate_events(
            API_KEY,
            site,
            start_time,
            end_time,
            period,
            voltage,
            default_battery_power,
            force_feasible=force_feasible,
        )
        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)
        quick_charge_obj = [
            ObjectiveComponent(quick_charge),
            ObjectiveComponent(equal_share, 1e-12),
        ]

        cls.alg = AdaptiveSchedulingAlgorithm(
            quick_charge_obj, solver=cp.ECOS, quantize=True, reallocate=True
        )
        cls.sim = acnsim.Simulator(
            cn, cls.alg, events, start_time, period=period, verbose=False
        )
        cls.sim.run()


class TestACACaltechOneDayUninterrupted(TestACACaltechSingleDayQuantized):
    @classmethod
    def setUpClass(cls) -> None:
        API_KEY = "DEMO_TOKEN"
        tz = pytz.timezone("America/Los_Angeles")
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = "caltech"
        basic_evse = False
        start = "9-01-2018"
        end = "9-02-2018"
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, "%m-%d-%Y"))
        end_time = tz.localize(datetime.strptime(end, "%m-%d-%Y"))

        events = acnsim.acndata_events.generate_events(
            API_KEY,
            site,
            start_time,
            end_time,
            period,
            voltage,
            default_battery_power,
            force_feasible=force_feasible,
        )
        cn = acnsim.network.sites.caltech_acn(
            basic_evse=basic_evse, voltage=voltage, transformer_cap=60
        )
        quick_charge_obj = [
            ObjectiveComponent(quick_charge),
            ObjectiveComponent(equal_share, 1e-12),
        ]

        cls.alg = AdaptiveSchedulingAlgorithm(
            quick_charge_obj, solver=cp.ECOS, quantize=True, uninterrupted_charging=True
        )
        cls.sim = acnsim.Simulator(
            cn, cls.alg, events, start_time, period=period, verbose=False
        )
        cls.sim.run()

    @classmethod
    def test_all_uninterrupted(cls):
        def uninterrupted(schedule):
            zero_found = False
            for r in schedule:
                if r == 0:
                    zero_found = True
                if r > 0 and zero_found:
                    return False
            return True

        station_ids = cls.sim.network.station_ids

        for ev in cls.sim.ev_history.values():
            i = station_ids.index(ev.station_id)
            pilot = cls.sim.pilot_signals[i, ev.arrival : ev.departure]
            assert uninterrupted(pilot)


# -------------------------------------------------------------
#  Offline Algorithm
# -------------------------------------------------------------


def test_internal_schedule_to_schedule_output():
    alg = AdaptiveChargingAlgorithmOffline([], acnsim.EventQueue())
    alg.internal_schedule = {
        "PS-1": [1, 2, 3, 4, 5],
        "PS-2": [0, 6, 7, 8, 9],
        "PS-3": [0, 0, 0, 10, 11],
    }
    alg.session_ids = {"1", "2", "3"}
    iface = Mock()
    iface.current_time = 3
    alg.register_interface(iface)
    ev1, ev2, ev3 = Mock(), Mock(), Mock()
    ev1.station_id, ev1.session_id = "PS-1", "1"
    ev2.station_id, ev2.session_id = "PS-2", "2"
    ev3.station_id, ev3.session_id = "PS-3", "3"
    schedule = alg.schedule([ev1, ev2, ev3])
    assert schedule["PS-1"][0] == 4
    assert schedule["PS-2"][0] == 8
    assert schedule["PS-3"][0] == 10


class TestAdaptiveChargingAlgorithmOfflineSingleEV(AdaptiveSchedulingAlgorithmBase):
    @classmethod
    def setUpClass(cls) -> None:
        timezone = pytz.timezone("America/Los_Angeles")
        start = timezone.localize(datetime(2018, 9, 1))
        period = 5  # minute
        voltage = 208  # volts

        cn = acnsim.ChargingNetwork()
        cn.register_evse(acnsim.get_evse_by_type("PS-1", "BASIC"), voltage, 0)
        cn.add_constraint(acnsim.Current("PS-1"), 100)

        batt = acnsim.Battery(100, 0, 7)
        ev = acnsim.EV(5, 5 + 12, voltage * 32 / 1000, "PS-1", "test", batt)
        events = acnsim.EventQueue([acnsim.PluginEvent(ev.arrival, ev)])

        quick_charge_obj = [
            ObjectiveComponent(quick_charge),
            ObjectiveComponent(equal_share, 1e-12),
        ]
        cls.alg = AdaptiveChargingAlgorithmOffline(quick_charge_obj, solver=cp.ECOS)
        cls.alg.register_events(events)
        cls.sim = acnsim.Simulator(
            cn, cls.alg, events, start, period=period, verbose=False
        )
        cls.alg.solve()
        cls.sim.run()

    @classmethod
    def test_internal_schedule_length(cls):
        assert len(cls.alg.internal_schedule["PS-1"]) == 17

    @classmethod
    def test_internal_schedule_matches_pilots(cls):
        assert np.allclose(
            cls.alg.internal_schedule["PS-1"], cls.sim.pilot_signals[0][:17], 1e-3
        )


class TestAdaptiveChargingAlgorithmOfflineCaltechSingleDay(
    AdaptiveSchedulingAlgorithmBase
):
    @classmethod
    def setUpClass(cls) -> None:
        API_KEY = "DEMO_TOKEN"
        tz = pytz.timezone("America/Los_Angeles")
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = "caltech"
        basic_evse = True
        start = "9-01-2019"
        end = "9-02-2019"
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, "%m-%d-%Y"))
        end_time = tz.localize(datetime.strptime(end, "%m-%d-%Y"))

        events = acnsim.acndata_events.generate_events(
            API_KEY,
            site,
            start_time,
            end_time,
            period,
            voltage,
            default_battery_power,
            force_feasible=force_feasible,
        )
        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)

        quick_charge_obj = [ObjectiveComponent(quick_charge)]
        cls.alg = AdaptiveChargingAlgorithmOffline(
            quick_charge_obj, solver=cp.ECOS, enforce_energy_equality=True
        )
        cls.alg.register_events(events)
        cls.sim = acnsim.Simulator(
            cn, cls.alg, events, start_time, period=period, verbose=False
        )
        cls.alg.solve()
        cls.sim.run()

    @classmethod
    def test_internal_schedule_energy_delivered_matches_requested(cls):
        internal_energy_delivered = (
            sum(sum(v) for v in cls.alg.internal_schedule.values())
            * 208
            / 1000
            * (5 / 60)
        )
        energy_requested = sum(
            ev.requested_energy for ev in cls.sim.ev_history.values()
        )
        assert np.allclose(internal_energy_delivered, energy_requested, atol=1e-2)


del AdaptiveSchedulingAlgorithmBase
#
#
# class TestAdaChargeLoadFlattening(TestCase):
#     def test_single_ev_feasible(self):
#         timezone = pytz.timezone('America/Los_Angeles')
#         start = timezone.localize(datetime(2018, 9, 1))
#         period = 5  # minute
#         voltage = 208  # volts
#
#         cn = acnsim.ChargingNetwork()
#         cn.register_evse(acnsim.get_evse_by_type('PS-1', 'BASIC'), voltage, 0)
#
#         batt = acnsim.Battery(100, 0, 7)
#         ev = acnsim.EV(5, 5+12, 6.6, 'PS-1', 'test', batt)
#         events = acnsim.EventQueue([acnsim.PluginEvent(ev.arrival, ev)])
#
#         alg = AdaChargeLoadFlattening(solver=cp.ECOS)
#
#         sim = acnsim.Simulator(cn, alg, events, start, period=period, verbose=False)
#         sim.run()
#
#         self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)
#
#     def test_caltech_testcase_feasible(self):
#         API_KEY = 'DEMO_TOKEN'
#         tz = pytz.timezone('America/Los_Angeles')
#         period = 5  # minutes
#         voltage = 208  # volts
#         default_battery_power = 32 * voltage / 1000  # kW
#         site = 'caltech'
#         basic_evse = True
#         start = '9-01-2018'
#         end = '9-02-2018'
#         force_feasible = True
#
#         start_time = tz.localize(datetime.strptime(start, '%m-%d-%Y'))
#         end_time = tz.localize(datetime.strptime(end, '%m-%d-%Y'))
#
#         events = acnsim.acndata_events.generate_events(API_KEY, site, start_time, end_time, period, voltage,
#                                                        default_battery_power, force_feasible=force_feasible)
#         cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)
#         alg = adacharge_load_flattening(solver=cp.ECOS)
#         sim = acnsim.Simulator(cn, alg, events, start_time, period=period, verbose=False)
#         sim.run()
#         self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)
#
#     def test_external_signal(self):
#         API_KEY = 'DEMO_TOKEN'
#         tz = pytz.timezone('America/Los_Angeles')
#         period = 5  # minutes
#         voltage = 208  # volts
#         default_battery_power = 32 * voltage / 1000  # kW
#         site = 'caltech'
#         basic_evse = True
#         start = '9-01-2018'
#         end = '9-02-2018'
#         force_feasible = True
#
#         start_time = tz.localize(datetime.strptime(start, '%m-%d-%Y'))
#         end_time = tz.localize(datetime.strptime(end, '%m-%d-%Y'))
#
#         events = acnsim.acndata_events.generate_events(API_KEY, site, start_time, end_time, period, voltage,
#                                                        default_battery_power, force_feasible=force_feasible)
#         cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)
#
#         external_sig = 20*np.random.rand(600)
#         alg = adacharge_load_flattening(external_signal=external_sig, solver=cp.ECOS)
#         sim = acnsim.Simulator(cn, alg, events, start_time, period=period, verbose=False)
#         sim.run()
#         self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)
