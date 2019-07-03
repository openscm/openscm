"""
Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999.
"""
import warnings
from typing import Union

import numpy as np

from ..core.parameters import HierarchicalName, ParameterInfo, ParameterType
from ..core.time import create_time_points
from ..core.units import _unit_registry
from ..errors import ParameterEmptyError
from ..models import PH99Model
from . import Adapter


class PH99(Adapter):
    """
    Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999.

    This one box model projects global-mean |CO2| concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of |CO2| alone.

    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    _hc_per_m2_approx = 1.34 * 10 ** 9 * _unit_registry("J / kelvin / m^2")
    """Approximate heat capacity per unit area (used to estimate rf2xco2)"""

    _base_time = np.datetime64("1750-01-01")
    """Base time. PH99 has no concept of datetimes so we make it up here"""

    _openscm_standard_parameter_mappings = {
        "Equilibrium Climate Sensitivity": "_ecs",
        "Radiative Forcing 2xCO2": "_rf2xco2",
        "Start Time": "_time_start",
        "Step Length": "_timestep",
        ("Emissions", "CO2"): "_emissions",
    }

    _openscm_output_mappings = {
        ("Emissions", "CO2"): "emissions",
        ("Atmospheric Concentrations", "CO2"): "concentrations",
        ("Cumulative Emissions", "CO2"): "cumulative_emissions",
        "Surface Temperature Increase": "temperatures",
    }

    _internal_timeseries_conventions = {
        "concentrations": "point",
        "cumulative_emissions": "point",
        "emissions": "point",
        "temperatures": "point",
    }

    @property
    def name(self):
        return "PH99"

    @property
    def _ecs(self):
        return self._parameters.scalar(
            "Equilibrium Climate Sensitivity", "delta_degC"
        ).value * _unit_registry("delta_degC")

    def _initialize_model(self) -> None:
        """
        Initialize the model.
        """
        self.model = PH99Model()  # pylint: disable=attribute-defined-outside-init
        imap = self._inverse_openscm_standard_parameter_mappings

        for att in dir(self.model):
            if not att.startswith(("_", "emissions_idx")):
                value = getattr(self.model, att)
                if callable(value):
                    continue

                try:
                    value.units
                except AttributeError:
                    self._add_parameter_view((self.name, att), value)
                    if att in imap:
                        openscm_name = imap[att]
                        self._add_parameter_view(openscm_name)
                if isinstance(value.magnitude, np.ndarray):
                    if att != "emissions":
                        continue
                    self._add_parameter_view(
                        (self.name, att),
                        unit=str(value.units),
                        timeseries_type=self._internal_timeseries_conventions[att],
                    )
                    if att in imap:
                        openscm_name = imap[att]
                        self._add_parameter_view(
                            openscm_name,
                            unit=str(value.units),
                            timeseries_type=self._internal_timeseries_conventions[att],
                        )
                else:
                    if att == "timestep":
                        units = "s"
                        mag = value.to(units).magnitude
                        if int(self.model.timestep.magnitude) != int(mag):
                            warnings.warn(
                                "Rounding {} timestep to nearest integer".format(
                                    self.name
                                )
                            )
                            value = int(mag) * _unit_registry(units)
                            self.model.timestep = value

                    self._add_parameter_view(
                        (self.name, att), value.magnitude, unit=str(value.units)
                    )
                    if att in imap:
                        openscm_name = imap[att]
                        if openscm_name in ("Start Time", "Stop Time", "Step Length"):
                            self._add_parameter_view(openscm_name)
                        else:
                            self._add_parameter_view(
                                openscm_name, unit=str(value.units)
                            )

        for openscm_name, self_att in self._openscm_standard_parameter_mappings.items():
            value = getattr(self, self_att)
            try:
                if isinstance(value.magnitude, np.ndarray):
                    model_name = self_att[1:]
                    self._add_parameter_view(
                        openscm_name,
                        unit=str(value.units),
                        timeseries_type=self._internal_timeseries_conventions[
                            model_name
                        ],
                    )
                else:
                    self._add_parameter_view(openscm_name, unit=str(value.units))
            except AttributeError:
                self._add_parameter_view(openscm_name)

        # set a stop time because PH99 model doesn't have one, probably shouldn't do this
        # implicitly...
        self._add_parameter_view("Stop Time")
        if self._parameter_views["Stop Time"].empty:
            self._parameter_views["Stop Time"].value = (
                self._start_time + 500 * self._period_length
            )

    def _get_view_iterator(self):
        view_iterator = self._parameter_views.items()
        view_iterator = sorted(view_iterator, key=lambda s: len(s[0]))
        generic_views = [
            v for v in view_iterator if not isinstance(v[1], dict) and v[1].parameter_type == ParameterType.GENERIC
        ]
        scalar_views_model = [
            v
            for v in view_iterator
            if not isinstance(v[1], dict) and v[1].parameter_type == ParameterType.SCALAR
            and isinstance(v[0], tuple)
            and len(v[0]) > 1
        ]
        scalar_views_openscm = [
            v
            for v in view_iterator
            if not isinstance(v[1], dict) and v[1].parameter_type == ParameterType.SCALAR
            and (not isinstance(v[0], tuple) or len(v[0]) == 1)
        ]
        scalar_views_openscm = sorted(  # ensure ECS considered before Radiative Forcing 2xCO2
            scalar_views_openscm, key=lambda s: s[0], reverse=False
        )
        other_views = [
            v
            for v in view_iterator
            if isinstance(v[1], dict) or v[1].parameter_type not in (ParameterType.GENERIC, ParameterType.SCALAR)
        ]
        return generic_views + scalar_views_model + scalar_views_openscm + other_views

    def _get_time_points(
        self, timeseries_type: Union[ParameterType, str]
    ) -> np.ndarray:
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

    @property
    def _start_time(self):
        try:
            return self._parameter_views["Start Time"].value
        except ParameterEmptyError:
            v = self._parameter_views[(self.name, "time_start")].value
            assert int(v) == v, "..."
            return self._base_time + np.timedelta64(int(v), "s")

    @property
    def _period_length(self):
        try:
            return self._parameter_views["Step Length"].value
        except ParameterEmptyError:
            v = self._parameter_views[(self.name, "timestep")].value
            assert int(v) == v, "..."
            return np.timedelta64(int(v), "s")

    @property
    def _timestep_count(self):
        stop_time = self._parameter_views["Stop Time"].value

        return (
            int((stop_time - self._start_time) / self._period_length) + 1
        )  # include self._stop_time

    def _timeseries_time_points_require_update(self):
        try:
            self._time_points
            self._time_points_for_averages
        except AttributeError:
            return True

        names_to_check = ["Start Time", "Stop Time", "Step Length"]
        for n in names_to_check:
            if self._parameter_views[n].version > self._parameter_versions[n]:
                return True
            if n in self._openscm_standard_parameter_mappings:
                model_n = (self.name, self._openscm_standard_parameter_mappings[n][1:])
                if (
                    self._parameter_views[model_n].version
                    > self._parameter_versions[model_n]
                ):
                    return True
        return False

    def _update_model(self, name: HierarchicalName, para: ParameterInfo) -> None:
        try:
            value = self._get_parameter_value(para)
            if name == "Stop Time":
                pass
            elif name in self._openscm_standard_parameter_mappings:
                self._set_model_para_from_openscm_para(name, value, para.unit)
            else:
                assert name[0] == self.name, "..."
                setattr(self.model, name[1], value * _unit_registry(para.unit))

        except ParameterEmptyError:
            pass

    def _set_model_para_from_openscm_para(self, openscm_name, value, unit):
        model_name = self._openscm_standard_parameter_mappings[openscm_name]
        if unit is not None:
            pv = value * _unit_registry(unit)
        else:
            pv = value

        setattr(self, model_name, pv)

    @property
    def _ecs(self):
        return self.model.mu * np.log(2) / self.model.alpha

    @_ecs.setter
    def _ecs(self, v):
        self._check_derived_paras(
            [(self.name, "alpha")],
            self._inverse_openscm_standard_parameter_mappings["_ecs"],
        )

        mu = self._parameter_views[(self.name, "mu")]
        mu = mu.value * _unit_registry(mu.unit)

        alpha = mu * np.log(2) / v
        self.model.alpha = alpha

    @property
    def _rf2xco2(self):
        return self.model.mu * self._hc_per_m2_approx

    @_rf2xco2.setter
    def _rf2xco2(self, v):
        self._check_derived_paras(
            [(self.name, "mu"), (self.name, "alpha")],
            self._inverse_openscm_standard_parameter_mappings["_rf2xco2"],
        )

        mu = v / self._hc_per_m2_approx
        alpha = mu * np.log(2) / self._ecs

        self.model.mu = mu
        self.model.alpha = alpha

    @property
    def _time_start(self):
        v = self.model.time_start.to("s").magnitude
        assert int(v) == v, "..."
        return self._base_time + np.timedelta64(int(v), "s")

    @_time_start.setter
    def _time_start(self, v):
        v = int((v - self._base_time).item().total_seconds())
        self.model.time_start = v * _unit_registry("s")

    @property
    def _timestep(self):
        v = self.model.timestep.to("s").magnitude
        assert int(v) == v, "..."
        return np.timedelta64(int(v), "s")

    @_timestep.setter
    def _timestep(self, v):
        import pdb

        pdb.set_trace()
        v = self.model.time_start.to("s").magnitude
        assert int(v) == v, "..."
        return self._base_time + np.timedelta64(int(v), "s")

    @property
    def _emissions(self):
        return self.model.emissions

    @_emissions.setter
    def _emissions(self, v):
        self.model.emissions = v

    def _initialize_openscm_standard_parameters(self):
        start_time = self._parameters.generic("Start Time")
        if start_time.empty:
            start_time.value = self._base_time + np.timedelta64(
                int(self.model.time_start.to("s").magnitude), "s"
            )

        ecs = self._parameters.scalar("Equilibrium Climate Sensitivity", "delta_degC")
        if ecs.empty:
            ecs.value = (
                (self.model.mu * np.log(2) / self.model.alpha)
                .to("delta_degC")
                .magnitude
            )
        else:
            self._set_model_parameter(
                "Equilibrium Climate Sensitivity",
                ecs.value * _unit_registry("delta_degC"),
            )

        rf2xco2 = self._parameters.scalar("Radiative Forcing 2xCO2", "W/m^2")
        if rf2xco2.empty:
            rf2xco2.value = (
                (self.model.mu * self._hc_per_m2_approx).to("W/m^2").magnitude
            )
        else:
            self._set_model_parameter(
                "Radiative Forcing 2xCO2", rf2xco2.value * _unit_registry("W/m^2")
            )

    def _reset(self) -> None:
        if (
            self._parameter_views[(self.name, "emissions")].empty
            and self._parameter_views[("Emissions", "CO2")].empty
        ):
            raise ParameterEmptyError(
                "{} requires ('Emissions', 'CO2') in order to run".format(self.name)
            )

    def _shutdown(self) -> None:
        pass

    def _run(self) -> None:
        self.model.initialise_timeseries()
        self.model.run()

        imap = {v: k for k, v in self._openscm_output_mappings.items()}
        self._inverse_openscm_standard_parameter_mappings
        for att in dir(self.model):
            # all time parameters captured in parameterset output
            if not att.startswith(("_", "time", "emissions_idx")):
                value = getattr(self.model, att)
                if callable(value) or not isinstance(value.magnitude, np.ndarray):
                    continue

                self._output.timeseries(
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
        self.model.initialise_timeseries()
        self.model.step()
        self._current_time = self._parameters.generic(
            "Start Time"
        ).value + np.timedelta64(int(self.model.time_current.to("s").magnitude), "s")
        # TODO: update output
