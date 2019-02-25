from openscm.adapters.dice import DICE

from base import _AdapterTester


class TestMyAdapter(_AdapterTester):
    tadapter = DICE

    # if necessary, you can extend the tests e.g.
    @classmethod
    def test_run(cls, test_adapter, test_config_paraset, test_drivers_core):
        super().test_run(test_adapter, test_config_paraset, test_drivers_core)
        # some specific test of your adapter here

    @classmethod
    def test_my_special_feature(cls, test_adapter):
        # test some special feature of your adapter class
        pass
