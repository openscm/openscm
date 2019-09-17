import os.path

import numpy as np
import pymagicc.core
import pymagicc.io
from typing import TYPE_CHECKING, Dict, Sequence, Union

from . import Adapter
from ..core.parameters import HierarchicalName, ParameterInfo, ParameterType, HIERARCHY_SEPARATOR
from ..errors import ParameterEmptyError

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
        "Stop Time": "endyear",
    }

    _openscm_output_mappings = {
        "Surface Temperature Increase": "Surface Temperature",
    }

    _internal_timeseries_conventions = {
        "Atmospheric Concentrations": "point",
        "Emissions": "average",
        "Temperatures": "point",
    }

    _units = {
        "core_climatesensitivity": "delta_degC",
        "core_delq2xco2": "W/m^2",
    }

    _write_out_emissions = False
    """bool: do emissions need to be written to disk?"""

    _run_kwargs = {}
    """dict: kwargs to be passed to the MAGICC run call"""

    @property
    def name(self):
        """
        Name of the model as used in OpenSCM parameters
        """
        return "MAGICC6"

    def _initialize_model(self) -> None:
        self.model = pymagicc.core.MAGICC6()
        self.model.create_copy()
        for nml_name, nml in self.model.default_config.items():
            for para, value in nml.items():
                if para in self._units:
                    self._initialize_scalar_view((self.name, para), value, self._units[para])
                else:
                    self._initialize_generic_view((self.name, para), value)

        for o_name, magicc6_name in self._openscm_standard_parameter_mappings.items():
            value = self._get_magcfg_default_value(magicc6_name)
            if magicc6_name in self._units:
                self._add_parameter_view(o_name, unit=self._units[magicc6_name])
            else:
                self._add_parameter_view(o_name)

        scen_emms = pymagicc.io.MAGICCData(os.path.join(self.model.run_dir, "RCP26.SCEN")).filter(region="World")
        for _, (emms, unit) in scen_emms.meta[["variable", "unit"]].drop_duplicates().iterrows():
            openscm_name = tuple(emms.split(HIERARCHY_SEPARATOR))
            self._initialize_timeseries_view(openscm_name, unit)

        # hack to initialise input timeseries too, have to think through better in
        # future...

    def _initialize_generic_view(self, full_name, value):
        self._add_parameter_view(full_name, value)
        model_name = full_name[1]
        imap = self._inverse_openscm_standard_parameter_mappings
        if model_name in imap:
            openscm_name = imap[model_name]
            self._add_parameter_view(openscm_name)

    def _initialize_scalar_view(self, full_name, value, unit):
        model_name = full_name[1]
        imap = self._inverse_openscm_standard_parameter_mappings

        self._add_parameter_view(full_name, value, unit=unit)
        if model_name in imap:
            openscm_name = imap[model_name]
            if openscm_name in ("Start Time", "Stop Time"):
                self._add_parameter_view(openscm_name)
            else:
                self._add_parameter_view(openscm_name, unit=unit)

    def _initialize_timeseries_view(self, full_name, unit):
        top_key = full_name[0]
        self._add_parameter_view(
            full_name,
            unit=unit,
            timeseries_type=self._internal_timeseries_conventions[top_key],
        )

    def _get_magcfg_default_value(self, magicc6_name):
        if magicc6_name in ("startyear", "endyear", "stepsperyear"):
            return self.model.default_config["nml_years"][magicc6_name]

        return self.model.default_config["nml_allcfgs"][magicc6_name]

    def _shutdown(self) -> None:
        self.model.remove_temp_copy()

    def _get_time_points(
        self, timeseries_type: Union[ParameterType, str]
    ) -> np.ndarray:
        # import pdb
        # pdb.set_trace()
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
        pass
        # import pdb
        # pdb.set_trace()

    def _set_model_from_parameters(self):
        super()._set_model_from_parameters()

        if self._write_out_emissions:
            import pdb
            pdb.set_trace()
            # dump out emissions to file now
            self._write_out_emissions = False

    def _update_model(self, name: HierarchicalName, para: ParameterInfo) -> None:
        value = self._get_parameter_value(para)
        if name in self._openscm_standard_parameter_mappings:
            self._set_model_para_from_openscm_para(name, value)
        else:
            if name[0] != self.name:
                # emergency valve for now, must be smarter way to handle this
                raise ValueError("How did non-MAGICC6 parameter end up here?")

            timeseries_types  = (ParameterType.AVERAGE_TIMESERIES, ParameterType.POINT_TIMESERIES)
            if para.parameter_type in timeseries_types:
                self._write_out_emissions = True
            self._run_kwargs[name[1]] = value

    def _set_model_para_from_openscm_para(self, openscm_name, value):
        magicc6_name = self._openscm_standard_parameter_mappings[openscm_name]

        if magicc6_name in ("startyear", "endyear"):
            self._run_kwargs[magicc6_name] = value.astype(object).year
        else:
            self._run_kwargs[magicc6_name] = value

    def _reset(self) -> None:
        pass
        # import pdb
        # pdb.set_trace()

    def _run(self) -> None:
        res = self.model.run()

        import pdb
        pdb.set_trace()
        imap = {v: k for k, v in self._openscm_output_mappings.items()}
        for att in dir(self.model):
            # all time parameters captured in parameterset output
            if not att.startswith(("_", "time", "emissions_idx")):
                value = getattr(self.model, att)
                if callable(value) or not isinstance(value.magnitude, np.ndarray):
                    continue

                self._output.timeseries(  # type: ignore
                    imap[att],
                    str(value.units),
                    time_points=self._get_time_points(
                        self._internal_timeseries_conventions[att]
                    ),
                    region="World",
                    timeseries_type=self._internal_timeseries_conventions[att],
                ).values = value.magnitude

        ecs = (self.model.mu * np.log(2) / self.model.alpha).to("K")
        self._output.scalar(
            ("Equilibrium Climate Sensitivity",), str(ecs.units), region=("World",)
        ).value = ecs.magnitude

        rf2xco2 = self.model.mu * self._hc_per_m2_approx
        self._output.scalar(
            ("Radiative Forcing 2xCO2",), str(rf2xco2.units), region=("World",)
        ).value = rf2xco2.magnitude

    def _step(self) -> None:
        raise NotImplementedError

    @property
    def _start_time(self):
        st = super()._start_time
        if isinstance(st, (float, int)):
            if int(st) != st:
                raise ValueError("('MAGICC6', 'startyear') should be an integer")
            return np.datetime64("{}-01-01".format(int(st)))
        return st

    @property
    def _end_time(self):
        try:
            return self._parameter_views["Stop Time"].value
        except ParameterEmptyError:
            et = self._parameter_views[
                (self.name, self._openscm_standard_parameter_mappings["Stop Time"])
            ].value
            if isinstance(et, (int, float)):
                if int(et) != et:
                    raise ValueError("('MAGICC6', 'endyear') should be an integer")
            return np.datetime64("{}-01-01".format(int(et)))

    @property
    def _timestep_count(self):
        # MAGICC6 is always run with yearly drivers, the `stepsperyear` parameter is
        # internal only so can be ignored
        return self._end_time.astype(object).year - self._start_time.astype(object).year + 1
