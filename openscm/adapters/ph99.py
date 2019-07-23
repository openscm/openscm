"""
Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999.
"""
import warnings
from typing import TYPE_CHECKING, Dict, Sequence

import numpy as np

from ..core.parameters import HierarchicalName, ParameterInfo
from ..core.units import _unit_registry
from ..errors import ParameterEmptyError
from ..models import PH99Model
from . import AdapterConstantTimestep

if TYPE_CHECKING:  # pragma: no cover
    import pint


class PH99(AdapterConstantTimestep):
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

    _openscm_standard_parameter_mappings: Dict[Sequence[str], str] = {
        "Equilibrium Climate Sensitivity": "ecs",
        "Radiative Forcing 2xCO2": "rf2xco2",
        "Start Time": "time_start",
        "Step Length": "timestep",
        ("Emissions", "CO2"): "emissions",
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
        """
        Name of the model as used in OpenSCM parameters
        """
        return "PH99"

    def _initialize_model(self) -> None:
        """
        Initialize the model.
        """
        self.model = PH99Model()  # pylint: disable=attribute-defined-outside-init
        for att in dir(self.model):
            if not att.startswith(("_", "emissions_idx")):
                value = getattr(self.model, att)
                if callable(value):
                    continue

                try:
                    value.units
                except AttributeError:
                    self._initialize_generic_view((self.name, att), value)

                if isinstance(value.magnitude, np.ndarray):
                    self._initialize_timeseries_view((self.name, att), value)
                else:
                    self._initialize_scalar_view((self.name, att), value)

        for o_name, self_att in self._openscm_standard_parameter_mappings.items():
            value = getattr(self, self_att)
            try:
                if isinstance(value.magnitude, np.ndarray):
                    self._add_parameter_view(
                        o_name,
                        unit=str(value.units),
                        timeseries_type=self._internal_timeseries_conventions[self_att],
                    )
                else:
                    self._add_parameter_view(o_name, unit=str(value.units))
            except AttributeError:
                self._add_parameter_view(o_name)

        # set a stop time because PH99 model doesn't have one, probably shouldn't do this
        # implicitly...
        self._add_parameter_view("Stop Time")
        if self._parameter_views["Stop Time"].empty:
            self._parameter_views["Stop Time"].value = (  # type: ignore
                self._start_time + 500 * self._period_length
            )

    def _initialize_generic_view(self, full_name, value):
        self._add_parameter_view(full_name, value)
        model_name = full_name[1]
        imap = self._inverse_openscm_standard_parameter_mappings
        if model_name in imap:
            openscm_name = imap[model_name]
            self._add_parameter_view(openscm_name)

    def _initialize_scalar_view(self, full_name, value):
        model_name = full_name[1]
        imap = self._inverse_openscm_standard_parameter_mappings
        if model_name == "timestep":
            units = "s"
            mag = value.to(units).magnitude
            if int(self.model.timestep.magnitude) != int(mag):
                warnings.warn(
                    "Rounding {} timestep to nearest integer".format(self.name)
                )
                value = int(mag) * _unit_registry(units)
                self.model.timestep = value

        self._add_parameter_view(full_name, value.magnitude, unit=str(value.units))
        if model_name in imap:
            openscm_name = imap[model_name]
            if openscm_name in ("Start Time", "Stop Time", "Step Length"):
                self._add_parameter_view(openscm_name)
            else:
                self._add_parameter_view(openscm_name, unit=str(value.units))

    def _initialize_timeseries_view(self, full_name, value):
        model_name = full_name[1]
        imap = self._inverse_openscm_standard_parameter_mappings
        if model_name != "emissions":
            return
        self._add_parameter_view(
            full_name,
            unit=str(value.units),
            timeseries_type=self._internal_timeseries_conventions[model_name],
        )
        if model_name in imap:
            openscm_name = imap[model_name]
            self._add_parameter_view(
                openscm_name,
                unit=str(value.units),
                timeseries_type=self._internal_timeseries_conventions[model_name],
            )

    def _get_scalar_views_openscm(self, view_iterator):
        svs = super()._get_scalar_views_openscm(view_iterator)
        svs = sorted(  # ensure ECS considered before Radiative Forcing 2xCO2
            svs, key=lambda s: s[0], reverse=False
        )
        return svs

    @property
    def _start_time(self):
        st = super()._start_time
        if isinstance(st, float):
            if int(st) != st:
                raise ValueError("('PH99', 'time_start') should be an integer")
            diff = np.timedelta64(  # pylint: disable=too-many-function-args
                int(st), "s"
            )
            return self._base_time + diff
        return st

    @property
    def _period_length(self):
        pl = super()._period_length
        if isinstance(pl, float):
            if int(pl) != pl:
                raise ValueError("('PH99', 'timestep') should be an integer")
            return np.timedelta64(  # pylint: disable=too-many-function-args
                int(pl), "s"
            )
        return np.timedelta64(pl, "s")  # pylint: disable=too-many-function-args

    def _update_model(self, name: HierarchicalName, para: ParameterInfo) -> None:
        value = self._get_parameter_value(para)
        if name == "Stop Time":
            pass
        elif name in self._openscm_standard_parameter_mappings:
            self._set_model_para_from_openscm_para(name, value, para.unit)
        else:
            if name[0] != self.name:
                # emergency valve for now, must be smarter way to handle this
                raise ValueError("How did non-PH99 parameter end up here?")
            setattr(self.model, name[1], value * _unit_registry(para.unit))

    def _set_model_para_from_openscm_para(self, openscm_name, value, unit):
        model_name = self._openscm_standard_parameter_mappings[openscm_name]
        if unit is not None:
            pv = value * _unit_registry(unit)
        else:
            pv = value

        setattr(self, model_name, pv)

    @property
    def ecs(self) -> "pint.Quantity":
        """
        Equilibrium climate sensitivity
        """
        return self.model.mu * np.log(2) / self.model.alpha

    @ecs.setter
    def ecs(self, v):
        self._check_derived_paras(
            [(self.name, "alpha")],
            self._inverse_openscm_standard_parameter_mappings["ecs"],
        )

        mu = self._parameter_views[(self.name, "mu")]
        mu = mu.value * _unit_registry(mu.unit)

        alpha = mu * np.log(2) / v
        self.model.alpha = alpha

    @property
    def rf2xco2(self) -> "pint.Quantity":
        """
        Radiative forcing due to a doubling of atmospheric |CO2| concentrations
        """
        return self.model.mu * self._hc_per_m2_approx

    @rf2xco2.setter
    def rf2xco2(self, v):
        self._check_derived_paras(
            [(self.name, "mu"), (self.name, "alpha")],
            self._inverse_openscm_standard_parameter_mappings["rf2xco2"],
        )

        mu = v / self._hc_per_m2_approx
        alpha = mu * np.log(2) / self.ecs

        self.model.mu = mu
        self.model.alpha = alpha

    @property
    def time_start(self) -> np.datetime64:
        """
        Start time of the run
        """
        v = self.model.time_start.to("s").magnitude
        if int(v) != v:
            raise ValueError("_time_start should be an integer")
        diff = np.timedelta64(int(v), "s")  # pylint: disable=too-many-function-args
        return self._base_time + diff

    @time_start.setter
    def time_start(self, v):
        v = (v - self._base_time).item().total_seconds()
        if int(v) != v:
            warnings.warn("Rounding {} time_start to nearest integer".format(self.name))
        self.model.time_start = int(v) * _unit_registry("s")

    @property
    def timestep(self) -> np.timedelta64:
        """
        Timestep of the run
        """
        v = self.model.timestep.to("s").magnitude
        if int(v) != v:
            raise ValueError("_timestep should be an integer")
        return np.timedelta64(int(v), "s")  # pylint: disable=too-many-function-args

    @timestep.setter
    def timestep(self, v):
        if int(v.magnitude) != v.magnitude:
            raise ValueError("_timestep should be an integer")
        self.model.timestep = v.to("s")

    @property
    def emissions(self):
        """
        |CO2| emissions driving this run
        """
        return self.model.emissions

    @emissions.setter
    def emissions(self, v):
        self.model.emissions = v

    def _reset(self) -> None:
        if (
            self._parameter_views[(self.name, "emissions")].empty
            and self._parameter_views[("Emissions", "CO2")].empty
        ):
            raise ParameterEmptyError(
                "{} requires ('Emissions', 'CO2') in order to run".format(self.name)
            )
        self.model.initialise_timeseries()
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

    def _shutdown(self) -> None:
        pass

    def _run(self) -> None:
        self.model.initialise_timeseries()
        self.model.run()

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
        self.model.initialise_timeseries()
        self.model.step()
        self._current_time = self._parameters.generic(
            "Start Time"
        ).value + np.timedelta64(  # pylint: disable=too-many-function-args
            int(self.model.time_current.to("s").magnitude), "s"
        )
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
