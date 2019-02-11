import progressbar


from .core import Core
from .highlevel import (
    OpenSCM,
    ScmDataFrame,
    convert_scmdataframe_to_core,
    convert_core_to_scmdataframe,
    convert_config_dict_to_parameter_set,
)
from .adapters import get_adapter
from .constants import ONE_YEAR_IN_S_INTEGER


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def run(drivers, model_configurations):
    assert isinstance(
        model_configurations, dict
    ), "model_configurations must be a dictionary"
    for climate_model, configurations in progressbar.progressbar(
        model_configurations.items()
    ):
        print("running {}\n".format(climate_model))
        runner = get_adapter(climate_model)()
        runner.initialize()
        for (scenario, model), sdf in progressbar.progressbar(
            drivers.timeseries().groupby(["scenario", "model"])
        ):
            print("running {}".format(scenario))
            parameter_set_scenario = convert_scmdataframe_to_core(ScmDataFrame(sdf))
            runner.set_drivers(parameter_set_scenario)
            for i, config in progressbar.progressbar(enumerate(configurations)):
                parameter_set_config = convert_config_dict_to_parameter_set(config)
                runner.set_config(parameter_set_config)
                config_results = runner.run()
                config_results = convert_core_to_scmdataframe(
                    config_results,
                    period_length=ONE_YEAR_IN_S_INTEGER,  # should uncode this hard coding
                    model=model,
                    scenario=scenario,
                    climate_model=climate_model,
                )
                try:
                    import pdb
                    pdb.set_trace()
                    results.append(config_results, inplace=True)
                except NameError:
                    results = config_results

    return results
