from openscm.adapters.magicc6 import MAGICC6

from base import _AdapterTester


class TestMAGICC6(_AdapterTester):
    tadapter = MAGICC6

    # if necessary, you can extend the tests e.g.
    def test_run(self, test_adapter, test_run_parameters):
        super().test_run(test_adapter, test_run_parameters)
        # TODO some specific test of your adapter here

    # def test_my_special_feature(self, test_adapter):
    #     # TODO test some special feature of your adapter class
