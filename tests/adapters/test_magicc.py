# Example file, would look something like this
from openscm.adapters import MAGICC6


from .base import _AdapterTester


class MAGICCTester(_AdapterTester):
    tadapter = MAGICC6
    # here you extend tests as necessary e.g.
    def test_run(self, test_adapter, test_config_paraset, test_drivers_core):
        super().test_run(test_adapter, test_config_paraset, test_drivers_core)
        # some MAGICC specific test here
