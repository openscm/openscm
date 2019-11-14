"""
Adapter for MAGICC6 (Meinshausen et al. ACP 2011)
"""
import os.path

import numpy as np
import pymagicc.core
import pymagicc.io

from ...core.parameters import HIERARCHY_SEPARATOR
from ...core.parameterset import ParameterSet
from .base import _MAGICCBase


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

    _pymagicc_class = pymagicc.core.MAGICC6
    """
    :cls:`pymagicc.core.MAGICCBase`: Pymagicc class which provides the MAGICC adapter
    """

    def __init__(self, input_parameters: ParameterSet, output_parameters: ParameterSet):
        """
        Initialize the MAGICC adapter as well as the Pymagicc class it will use.

        *Note:* as part of this process, all available model parameters are added to
        ``input_parameters`` (if they're not already there).

        Parameters
        ----------
        input_parameters
            Input parameter set to use

        output_parameters
            Output parameter set to use
        """

        self._run_kwargs = {}
        """dict: kwargs to be passed to the MAGICC run call"""

        self._write_out_emissions = False
        """bool: do emissions need to be written to disk?"""

        self.model = self._pymagicc_class()
        """:obj:`pymagicc.core.MAGICCBase`: instance of the Pymagicc class"""

        super().__init__(input_parameters, output_parameters)

    @property
    def name(self):
        """
        Name of the model as used in OpenSCM parameters
        """
        return "MAGICC6"

    def _initialize_model(self) -> None:
        self.model.create_copy()
        for _, nml in self.model.default_config.items():
            for para, value in nml.items():
                if para in self._units:
                    self._initialize_scalar_view(
                        (self.name, para), value, self._units[para]
                    )
                else:
                    self._initialize_generic_view((self.name, para), value)

        for o_name, magicc_name in self._openscm_standard_parameter_mappings.items():
            value = self._get_magcfg_default_value(magicc_name)
            if magicc_name in self._units:
                self._add_parameter_view(o_name, unit=self._units[magicc_name])
            else:
                self._add_parameter_view(o_name)

        scen_emms = pymagicc.io.MAGICCData(
            os.path.join(self.model.run_dir, "RCP26.SCEN")
        ).filter(region="World")
        for _, (emms, unit) in (
            scen_emms.meta[["variable", "unit"]].drop_duplicates().iterrows()
        ):
            openscm_name = tuple(emms.split(HIERARCHY_SEPARATOR))
            self._initialize_timeseries_view(openscm_name, unit)

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
            },
        )

        scen.write(
            os.path.join(self.model.run_dir, "PYMAGICC.SCEN"),
            magicc_version=self.model.version,
        )
        self.model.update_config(file_emissionscenario="PYMAGICC.SCEN")
