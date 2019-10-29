"""
OpenSCM - unified access to simple climate models.
"""
from typing import List, Union

import numpy as np
import tqdm
from pyam import IamDataFrame

from ._version import get_versions
from .core import OpenSCM  # noqa: F401
from .scmdataframe import ScmDataFrame, OpenScmDataFrame, convert_openscm_to_scmdataframe, df_append

__version__: str = get_versions()["version"]
del get_versions


def run(
    emissions: Union[ScmDataFrame, OpenScmDataFrame, IamDataFrame],
    climate_models: List[str],
    output_time_points: np.ndarray = [
        np.datetime64("{}-01-01".format(y)) for y in range(1765, 2101)
    ],
) -> Union[OpenScmDataFrame, IamDataFrame]:
    """
    Run a series of emissions scenarios

    Parameters
    ----------
    emissions
        Dataframe holding the emissions scenarios to run

    climate_models
        Climate models to run

    output_time_points
        The points on which to report the results of the runs
    """
    if isinstance(emissions, IamDataFrame):
        runner = ScmDataFrame(emissions)
    else:
        runner = OpenScmDataFrame(emissions)

    results = []
    for climate_model in tqdm.tqdm_notebook(climate_models, desc="Climate Models"):
        unique_model_scens = runner[["model", "scenario"]].drop_duplicates()
        for i, label in tqdm.tqdm_notebook(
            unique_model_scens.iterrows(),
            total=len(unique_model_scens),
            leave=True,
            desc=climate_model,
        ):
            label = label.to_dict()
            model = label["model"]
            scenario = label["scenario"]
            run_df = runner.filter(model=model, scenario=scenario)

            ps = OpenScmDataFrame(run_df).to_parameterset()
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
    output_scmdf = OpenScmDataFrame(
        df_append(results).timeseries().reset_index().fillna(-999)
    )

    if isinstance(emissions, IamDataFrame):
        return output_scmdf.to_iamdataframe()
    return output_scmdf
