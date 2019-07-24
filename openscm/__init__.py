"""
OpenSCM - unified access to simple climate models.
"""
from typing import Union, List

import numpy as np
import tqdm

from ._version import get_versions
from .core import OpenSCM  # noqa: F401
from .scmdataframe import ScmDataFrame, df_append, convert_openscm_to_scmdataframe
from pyam import IamDataFrame

__version__: str = get_versions()["version"]
del get_versions



def run(
    emissions: Union[ScmDataFrame, IamDataFrame],
    climate_models: List[str],
    output_time_points: np.ndarray = [np.datetime64("{}-01-01".format(y)) for y in range(1765, 2101)]
) -> Union[ScmDataFrame, IamDataFrame]:
    if isinstance(emissions, IamDataFrame):
        runner = ScmDataFrame(emissions)
    else:
        runner = emissions.copy()

    results = []
    for climate_model in climate_models:
        unique_model_scens = runner[["model", "scenario"]].drop_duplicates()
        for i, label in tqdm.tqdm(unique_model_scens.iterrows(), total=len(unique_model_scens), leave=True, desc=climate_model):
            label = label.to_dict()
            model = label["model"]
            scenario = label["scenario"]
            run_df = runner.filter(model=model, scenario=scenario)

            if climate_model == "PH99":
                # massive hack required only because of lack of point to average conversion
                run_df = run_df.timeseries().reset_index()
                run_df["parameter_type"] = "point"
                run_df = ScmDataFrame(run_df)

            ps = ScmDataFrame(run_df).to_parameterset()
            ps.generic("Start Time").value = np.datetime64(run_df["time"].min())
            ps.generic("Stop Time").value = np.datetime64(run_df["time"].max())
            # not how this should be done, should initialise earlier...
            climate_runner = OpenSCM(climate_model, ps)
            climate_runner.run()
            output_scmdf = convert_openscm_to_scmdataframe(
                climate_runner.output,
                time_points=output_time_points,
                climate_model=climate_model,
                model=model,
                scenario=scenario,
            )
            results.append(output_scmdf.timeseries())

    # hack required to ensure IamDataFrame keeps all output
    output_scmdf = ScmDataFrame(df_append(results).timeseries().reset_index().fillna(-999))

    if isinstance(emissions, IamDataFrame):
        return output_scmdf.to_iamdataframe()
    return output_scmdf

