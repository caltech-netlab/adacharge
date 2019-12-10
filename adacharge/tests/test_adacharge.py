from unittest import TestCase
import numpy as np

import pytz
from datetime import datetime
import cvxpy as cp
from acnportal import acnsim
from acnportal.signals import tariffs
from adacharge import AdaCharge, AdaChargeProfitMax, AdaChargeLoadFlattening


class TestAdaCharge(TestCase):
    def test_single_ev_feasible(self):
        timezone = pytz.timezone('America/Los_Angeles')
        start = timezone.localize(datetime(2018, 9, 1))
        period = 5  # minute
        voltage = 208  # volts

        cn = acnsim.ChargingNetwork()
        cn.register_evse(acnsim.get_evse_by_type('PS-1', 'BASIC'), voltage, 0)

        batt = acnsim.Battery(100, 0, 7)
        ev = acnsim.EV(5, 5+12, 6.6, 'PS-1', 'test', batt)
        events = acnsim.EventQueue([acnsim.PluginEvent(ev.arrival, ev)])

        alg = AdaCharge(solver=cp.ECOS)

        sim = acnsim.Simulator(cn, alg, events, start, period=period, verbose=False)
        sim.run()

        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)

    def test_caltech_testcase_feasible(self):
        API_KEY = 'DEMO_TOKEN'
        tz = pytz.timezone('America/Los_Angeles')
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = 'caltech'
        basic_evse = True
        start = '9-01-2018'
        end = '9-02-2018'
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, '%m-%d-%Y'))
        end_time = tz.localize(datetime.strptime(end, '%m-%d-%Y'))

        events = acnsim.acndata_events.generate_events(API_KEY, site, start_time, end_time, period, voltage,
                                                       default_battery_power, force_feasible=force_feasible)
        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)
        alg = AdaCharge(solver=cp.ECOS)
        sim = acnsim.Simulator(cn, alg, events, start_time, period=period, verbose=False)
        sim.run()
        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)


class TestAdaChargeOffline(TestCase):
    def test_single_ev_feasible(self):
        timezone = pytz.timezone('America/Los_Angeles')
        start = timezone.localize(datetime(2018, 9, 1))
        period = 5  # minute
        voltage = 208  # volts

        cn = acnsim.ChargingNetwork()
        cn.register_evse(acnsim.get_evse_by_type('PS-1', 'BASIC'), voltage, 0)

        batt = acnsim.Battery(100, 0, 7)
        ev = acnsim.EV(5, 5+12, voltage*32/1000, 'PS-1', 'test', batt)
        events = acnsim.EventQueue([acnsim.PluginEvent(ev.arrival, ev)])

        alg = AdaCharge(offline=True, energy_equality=True, events=events)

        sim = acnsim.Simulator(cn, alg, events, start, period=period, verbose=False)
        sim.run()

        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)
        np.testing.assert_almost_equal(sum(sum(v) for v in alg.internal_schedule.values()), sim.charging_rates.sum(), decimal=4)

    def test_caltech_testcase_feasible(self):
        API_KEY = 'DEMO_TOKEN'
        tz = pytz.timezone('America/Los_Angeles')
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = 'caltech'
        basic_evse = True
        start = '1-01-2019'
        end = '2-01-2019'
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, '%m-%d-%Y'))
        end_time = tz.localize(datetime.strptime(end, '%m-%d-%Y'))

        events = acnsim.acndata_events.generate_events(API_KEY, site, start_time, end_time, period, voltage,
                                                       default_battery_power, force_feasible=force_feasible)
        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)
        alg = AdaCharge(energy_equality=True, solver=cp.MOSEK, offline=True, events=events)
        sim = acnsim.Simulator(cn, alg, events, start_time, period=period, verbose=False)
        sim.run()
        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)
        np.testing.assert_allclose(sum(sum(v) for v in alg.internal_schedule.values()), sim.charging_rates.sum(), rtol=1e-04)


class TestAdaChargeProfitMax(TestCase):
    def test_single_ev_feasible(self):
        timezone = pytz.timezone('America/Los_Angeles')
        start = timezone.localize(datetime(2018, 9, 1))
        period = 5  # minute
        voltage = 208  # volts

        cn = acnsim.ChargingNetwork()
        cn.register_evse(acnsim.get_evse_by_type('PS-1', 'BASIC'), voltage, 0)

        batt = acnsim.Battery(100, 0, 7)
        ev = acnsim.EV(5, 5+12, 6.6, 'PS-1', 'test', batt)
        events = acnsim.EventQueue([acnsim.PluginEvent(ev.arrival, ev)])

        alg = AdaChargeProfitMax(1000, solver=cp.ECOS)

        signals = {'tariff': tariffs.TimeOfUseTariff('sce_tou_ev_4_march_2019')}
        sim = acnsim.Simulator(cn, alg, events, start, period=period, verbose=False, signals=signals)
        sim.run()

        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)

    def test_caltech_testcase_feasible(self):
        API_KEY = 'DEMO_TOKEN'
        tz = pytz.timezone('America/Los_Angeles')
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = 'caltech'
        basic_evse = True
        start = '9-01-2018'
        end = '9-02-2018'
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, '%m-%d-%Y'))
        end_time = tz.localize(datetime.strptime(end, '%m-%d-%Y'))

        events = acnsim.acndata_events.generate_events(API_KEY, site, start_time, end_time, period, voltage,
                                                       default_battery_power, force_feasible=force_feasible)
        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)
        alg = AdaChargeProfitMax(1000, solver=cp.ECOS)
        signals = {'tariff': tariffs.TimeOfUseTariff('sce_tou_ev_4_march_2019')}
        sim = acnsim.Simulator(cn, alg, events, start_time, period=period, verbose=False, signals=signals)
        sim.run()
        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)


class TestAdaChargeLoadFlattening(TestCase):
    def test_single_ev_feasible(self):
        timezone = pytz.timezone('America/Los_Angeles')
        start = timezone.localize(datetime(2018, 9, 1))
        period = 5  # minute
        voltage = 208  # volts

        cn = acnsim.ChargingNetwork()
        cn.register_evse(acnsim.get_evse_by_type('PS-1', 'BASIC'), voltage, 0)

        batt = acnsim.Battery(100, 0, 7)
        ev = acnsim.EV(5, 5+12, 6.6, 'PS-1', 'test', batt)
        events = acnsim.EventQueue([acnsim.PluginEvent(ev.arrival, ev)])

        alg = AdaChargeLoadFlattening(solver=cp.ECOS)

        sim = acnsim.Simulator(cn, alg, events, start, period=period, verbose=False)
        sim.run()

        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)

    def test_caltech_testcase_feasible(self):
        API_KEY = 'DEMO_TOKEN'
        tz = pytz.timezone('America/Los_Angeles')
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = 'caltech'
        basic_evse = True
        start = '9-01-2018'
        end = '9-02-2018'
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, '%m-%d-%Y'))
        end_time = tz.localize(datetime.strptime(end, '%m-%d-%Y'))

        events = acnsim.acndata_events.generate_events(API_KEY, site, start_time, end_time, period, voltage,
                                                       default_battery_power, force_feasible=force_feasible)
        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)
        alg = AdaChargeLoadFlattening(solver=cp.ECOS)
        sim = acnsim.Simulator(cn, alg, events, start_time, period=period, verbose=False)
        sim.run()
        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)

    def test_external_signal(self):
        API_KEY = 'DEMO_TOKEN'
        tz = pytz.timezone('America/Los_Angeles')
        period = 5  # minutes
        voltage = 208  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = 'caltech'
        basic_evse = True
        start = '9-01-2018'
        end = '9-02-2018'
        force_feasible = True

        start_time = tz.localize(datetime.strptime(start, '%m-%d-%Y'))
        end_time = tz.localize(datetime.strptime(end, '%m-%d-%Y'))

        events = acnsim.acndata_events.generate_events(API_KEY, site, start_time, end_time, period, voltage,
                                                       default_battery_power, force_feasible=force_feasible)
        cn = acnsim.network.sites.caltech_acn(basic_evse=basic_evse, voltage=voltage)

        external_sig = 20*np.random.rand(600)
        alg = AdaChargeLoadFlattening(external_signal=external_sig, solver=cp.ECOS)
        sim = acnsim.Simulator(cn, alg, events, start_time, period=period, verbose=False)
        sim.run()
        self.assertGreater(acnsim.analysis.proportion_of_energy_delivered(sim), .9999)
