from unittest import TestCase
from unittest.mock import create_autospec

from acnportal.acnsim import Interface
import numpy as np

import pytz
from datetime import datetime
from acnportal import acnsim
from adacharge.post_processor import *
from acnportal import algorithms


class TestAdaCharge(TestCase):
    def setUp(self):
        timezone = pytz.timezone('America/Los_Angeles')
        start = timezone.localize(datetime(2018, 9, 1))
        period = 5  # minute
        events = acnsim.EventQueue()
        events.add_event(acnsim.RecomputeEvent(1))  # ACNSim currently throws an error when their are no events
        cn = acnsim.sites.auto_acn.simple_acn(['PS-{0}'.format(x) for x in range(10)], evse_type=acnsim.AV,
                                              aggregate_cap=15)
        self.sim = acnsim.Simulator(cn, algorithms.UncontrolledCharging(), events,
                                    start, period=period, verbose=False)
        self.active_evs = [acnsim.EV(0, 10, 15, 'PS-{0}'.format(x), '{0}'.format(x),
                                     acnsim.Battery(50, 0, 6.6)) for x in range(10)]

    def _general_properties(self, new_schedule):
        for sch in new_schedule:
            for s in new_schedule[sch]:
                self.assertIsInstance(s, int)
                self.assertGreaterEqual(s, 6)

    def test_no_change_necessary(self):
        iface = acnsim.Interface(self.sim)
        feasible_schedule = {'PS-{0}'.format(x): [6]*10 for x in range(10)}
        pp = AdaChargePostProcessor(iface)
        new_schedule = pp.process(feasible_schedule, self.active_evs)
        for evse_id in new_schedule:
            self.assertEqual(new_schedule[evse_id][0], feasible_schedule[evse_id][0])
        self._general_properties(new_schedule)
        self.assertTrue(iface.is_feasible(new_schedule))

    def test_input_schedule_floats(self):
        iface = acnsim.Interface(self.sim)
        float_schedule = {'PS-{0}'.format(x): [6.6]*10 for x in range(10)}
        pp = AdaChargePostProcessor(iface)
        new_schedule = pp.process(float_schedule, self.active_evs)
        for sch in new_schedule:
            for i, s in enumerate(new_schedule[sch]):
                self.assertIsInstance(s, int)
                self.assertGreaterEqual(s, 6)
        self.assertTrue(iface.is_feasible(new_schedule))

    def test_less_than_min(self):
        iface = acnsim.Interface(self.sim)
        float_schedule = {'PS-{0}'.format(x): [4.5]*10 for x in range(10)}
        pp = AdaChargePostProcessor(iface)
        new_schedule = pp.process(float_schedule, self.active_evs)
        for sch in new_schedule:
            for i, s in enumerate(new_schedule[sch]):
                self.assertIsInstance(s, int)
                self.assertGreaterEqual(s, 6)
        self.assertTrue(iface.is_feasible(new_schedule))

    def test_redistribution_with_limit(self):
        iface = acnsim.Interface(self.sim)
        float_schedule = {'PS-{0}'.format(x): [24]*10 if x < 3 else [0]*10 for x in range(10)}
        pp = AdaChargePostProcessor(iface)
        new_schedule = pp.process(float_schedule, self.active_evs)
        for sch in new_schedule:
            for i, s in enumerate(new_schedule[sch]):
                self.assertIsInstance(s, int)
                self.assertGreaterEqual(s, 6)
        self.assertTrue(iface.is_feasible(new_schedule))

    def test_redistribution_with_limit_integer(self):
        iface = acnsim.Interface(self.sim)
        float_schedule = {'PS-{0}'.format(x): [24]*10 if x < 3 else [0]*10 for x in range(10)}
        pp = AdaChargePostProcessor(iface, integer_program=True)
        new_schedule = pp.process(float_schedule, self.active_evs)
        for sch in new_schedule:
            for i, s in enumerate(new_schedule[sch]):
                self.assertIsInstance(s, int)
                self.assertGreaterEqual(s, 6)
        self.assertTrue(iface.is_feasible(new_schedule))

    def test_CC_8_A_min(self):
        timezone = pytz.timezone('America/Los_Angeles')
        start = timezone.localize(datetime(2018, 9, 1))
        period = 5  # minute
        events = acnsim.EventQueue()
        events.add_event(acnsim.RecomputeEvent(1))  # ACNSim currently throws an error when their are no events
        cn = acnsim.sites.auto_acn.simple_acn(['PS-{0}'.format(x) for x in range(10)], evse_type=acnsim.CC,
                                              aggregate_cap=20)
        self.sim = acnsim.Simulator(cn, algorithms.UncontrolledCharging(), events,
                                    start, period=period, verbose=False)
        self.active_evs = [acnsim.EV(0, 10, 15, 'PS-{0}'.format(x), '{0}'.format(x),
                                     acnsim.Battery(50, 0, 6.6)) for x in range(10)]

        iface = acnsim.Interface(self.sim)
        float_schedule = {'PS-{0}'.format(x): [4.5] * 10 for x in range(10)}
        pp = AdaChargePostProcessor(iface)
        new_schedule = pp.process(float_schedule, self.active_evs)
        for sch in new_schedule:
            for i, s in enumerate(new_schedule[sch]):
                self.assertIsInstance(s, int)
                self.assertGreaterEqual(s, 8)
        self.assertTrue(iface.is_feasible(new_schedule))

