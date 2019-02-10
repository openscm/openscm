from openscm import run
from openscm.scenarios import rcps
from openscm.units import unit_registry

from openscm.scmdataframebase import LongIamDataFrame
from openscm.adapters import get_adapter
from openscm.highlevel import (
    OpenSCM,
    ScmDataFrame,
    convert_scmdataframe_to_core,
    convert_core_to_scmdataframe,
    convert_config_dict_to_parameter_set,
)
from openscm.constants import ONE_YEAR_IN_S_INTEGER

rcp26 = rcps.filter(scenario="RCP26")

runner = get_adapter("PH99")()
runner.initialize()
parameter_set_scenario = convert_scmdataframe_to_core(
    ScmDataFrame(rcp26)
)
runner.set_drivers(parameter_set_scenario)
config_results = runner.run()
config_results = convert_core_to_scmdataframe(
    config_results,
    period_length=ONE_YEAR_IN_S_INTEGER,  # should uncode this hard coding
    model="a",
    scenario="b",
    climate_model="c",
)
