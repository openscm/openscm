import os.path

import numpy as np
import pymagicc.io

from .base import _MAGICCBase
from ...core.parameters import HIERARCHY_SEPARATOR

class MAGICC6(_MAGICCBase):
    """
    Adapter for the MAGICC model, version 6.

    The Model for the Assessment of Greenhouse Gas Induced Climate Change (MAGICC)
    projects atmospheric greenhouse gas concentrations, radiative forcing of
    greenhouse gases and aerosols, hemispheric and land/ocean surface temperatures and
    sea-level rise from projected emissions of greenhouse gases and aerosols (its
    historical emissions/concentrations can also be specified but this functionality
    is not yet provided).

    Further reference:
    Meinshausen, M., Raper, S. C. B., and Wigley, T. M. L.: Emulating coupled
    atmosphere-ocean and carbon cycle models with a simpler model, MAGICC6 – Part 1:
    Model description and calibration, Atmos. Chem. Phys., 11, 1417-1456,
    https://doi.org/10.5194/acp-11-1417-2011, 2011.
    """
    def _write_emissions_to_file(self):
        data = []
        time = self._get_time_points("point")
        variable = []
        region = []
        unit = []
        for k, v in self._parameter_views.items():
            if k[0] == "Emissions":
                data.append(v.values)
                variable.append(HIERARCHY_SEPARATOR.join(k))
                region.append(HIERARCHY_SEPARATOR.join(v.region))
                unit.append(v.unit)

        scen = pymagicc.io.MAGICCData(
            data=np.vstack(data).T,
            index=time,
            columns={
                "variable": variable,
                "region": region,
                "unit": unit,
                "model": "na",
                "scenario": "na",
                "todo": "SET",
            }
        )

        scen.write(
            os.path.join(self.model.run_dir, "PYMAGICC.SCEN"),
            magicc_version=self.model.version,
        )
        self.model.update_config(file_emissionscenario="PYMAGICC.SCEN")
