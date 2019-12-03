from unittest import TestCase
from unittest.mock import create_autospec

from acnportal.acnsim import Interface
import numpy as np

import pytz
from datetime import datetime
import cvxpy as cp
from acnportal import acnsim
from adacharge.adacharge import *


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
    def test_default_get_demand_charge(self):
        ada = AdaChargeProfitMax(0.4)
        iface = create_autospec(Interface)
        iface.get_demand_charge.return_value = 15.51
        self.assertEqual(ada.get_demand_charge(iface), 15.51)

    def test_custom_get_demand_charge(self):
        def custom_demand_charge_proxy(i):
            return 14.0
        ada = AdaChargeProfitMax(0.4, get_dc=custom_demand_charge_proxy)
        iface = create_autospec(Interface)
        iface.get_demand_charge.return_value = 15.51
        self.assertEqual(ada.get_demand_charge(iface), 14.0)

    # def test_obj_default_demand_charge(self):
    #     active_evs = []
    #     for i in range(5):
    #         ev = EV(0, 8, 10, 'PS-{0}'.format(i), '{0}'.format(i), Battery)
    #         active_evs.append(ev)
    #     ada = AdaChargeProfitMax(0.4)
    #     iface = create_autospec(Interface)
    #     iface.get_demand_charge.return_value = 15.51  # $/kW
    #     iface.get_prices.return_value = np.array([.1, .2, .3, .4, .5, .6, .7, .8])  # $/kWh
    #     iface.period = 5
    #     iface.current_time = 0
    #     iface.active_evs = active_evs
    #     ada.register_interface(iface)
    #     ada.run()


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
