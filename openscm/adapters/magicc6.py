from pymagicc.core import MAGICC6
from typing import TYPE_CHECKING, Dict, Sequence

from . import Adapter

YEAR = 365 * 24 * 60 * 60  # example time step length as used below

# Zeb's scribbles as he does this (may be helpful for later):
#   - MAGICC doesn't need to have a constant timestep, hence can't directly copy the
#     examples of DICE or PH99
#   - _initialize_model needs to do initalize (create copy of MAGICC)
#   - _shutdown needs to do the cleanup (delete copy of MAGICC)
#   - _set_model_from_parameters needs to re-write the input files
#   - reset needs to delete everything in `out` and cleanup the output parameters
#   - _run just calls the .run method
#   - _step is not implemented
#   - _get_time_points can just use the openscm parameters (only need to pass back
#       down at set_model_from_parameters calls)
#   - _update_model just calls MAGICC's update_config method
#   - name is MAGICC6 (as it's what is used in OpenSCM parameters)
#   - tests
#       - start by reading through tests of DICE and PH99 for inspiration of what to
#         test
#   - follow lead of PH99 for implementation as it has a separate model module


class MAGICC6(Adapter):
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
    _openscm_standard_parameter_mappings: Dict[Sequence[str], str] = {
        "Equilibrium Climate Sensitivity": "core_climatesensitivity",
        "Radiative Forcing 2xCO2": "core_delq2xco2",
        "Start Time": "startyear",
        "Step Length": "endyear",
    }

    _openscm_output_mappings = {
        "Surface Temperature Increase": "Surface Temperature",
    }

    _internal_timeseries_conventions = {
        "Atmospheric Concentrations": "point",
        "Emissions": "average",
        "Temperatures": "point",
    }

    @property
    def name(self):
        """
        Name of the model as used in OpenSCM parameters
        """
        return "MAGICC6"

    def _initialize_model(self) -> None:
        self.model = MAGICC6()
        self.model.create_copy()
        import pdb
        pdb.set_trace()
        self.model.default_config()

    def _shutdown(self) -> None:
        self.model.remove_temp_copy()

    def _get_time_points(
        self, timeseries_type: Union[ParameterType, str]
    ) -> np.ndarray:
        import pdb
        pdb.set_trace()
        if self._timeseries_time_points_require_update():

            def get_time_points(tt):
                return create_time_points(
                    self._start_time, self._period_length, self._timestep_count, tt
                )

            self._time_points = get_time_points("point")
            self._time_points_for_averages = get_time_points("average")

        return (
            self._time_points
            if timeseries_type in ("point", ParameterType.POINT_TIMESERIES)
            else self._time_points_for_averages
        )

    def _timeseries_time_points_require_update(self) -> bool:
        import pdb
        pdb.set_trace()
