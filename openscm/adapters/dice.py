"""
Adapter for the climate component from the Dynamic Integrated Climate-Economy (DICE)
model by William Nordhaus (DICE 2013).

Original source: https://sites.google.com/site/williamdnordhaus/dice-rice

http://www.econ.yale.edu/~nordhaus/homepage/homepage/DICE2013R_100413_vanilla.gms
http://www.econ.yale.edu/~nordhaus/homepage/homepage/DICE2016R-091916ap.gms

This implementation follows the original DICE code closely, especially regarding
variable naming. Original comments are marked by "Original:".
"""

from collections import namedtuple
from math import log2
from typing import Any, Optional, Tuple, Union

import numpy as np

from ..core.parameters import HierarchicalName, ParameterInfo, ParameterType
from ..core.time import create_time_points
from ..errors import ParameterEmptyError
from . import Adapter

YEAR = np.timedelta64(365, "D")  # pylint: disable=too-many-function-args

_default_start_time = np.datetime64("1000-01-01")
_default_stop_time = np.datetime64("3100-01-01")

MODEL_PARAMETER_DEFAULTS = {
    # Initial size of atmospheric CO2 pool
    #     Original: "Initial Concentration in atmosphere 2010 (GtC)"
    "mat0": (830.4, "GtC"),  # 851
    # Equilibrium size of atmospheric CO2 pool
    #     Original: "Equilibrium concentration atmosphere (GtC)"
    "mateq": (588, "GtC"),  # 588
    "mat_lower": (10, "GtC"),  # 10
    # Original: "Initial Concentration in lower strata 2010 (GtC)"
    "ml0": (10010, "GtC"),  # 1740
    # Original: "Equilibrium concentration in lower strata (GtC)"
    "mleq": (10000, "GtC"),  # 1720
    "ml_lower": (1000, "GtC"),  # 1000
    # Original: "Initial Concentration in upper strata 2010 (GtC)"
    "mu0": (1527, "GtC"),  # 460
    # Original: "Equilibrium concentration in upper strata (GtC)"
    "mueq": (1350, "GtC"),  # 360
    "mu_lower": (100, "GtC"),  # 100
    # Original: "Initial atmospheric temp change (C from 1900)"
    "tatm0": (0.8, "delta_degC"),  # 0.85
    "tatm_upper": (40, "delta_degC"),  # 12
    # Original: "Initial lower stratum temp change (C from 1900)"
    "tocean0": (0.0068, "delta_degC"),  # 0.0068
    "tocean_lower": (-1, "delta_degC"),  # -1
    "tocean_upper": (20, "delta_degC"),  # 20
    # Original: "Carbon cycle transition matrix"
    "b12": (0.0181, ""),  # 0.088; 0.12
    # Original: "Carbon cycle transition matrix"
    "b23": (0.00071, ""),  # 0.00250; 0.007
    # Original: "Climate equation coefficient for upper level"
    "c1": (0.0222, "delta_degC*m^2/W"),  # 0.098; 0.1005
    # Original: "Transfer coefficient upper to lower stratum"
    "c3": (0.09175, "W/m^2/delta_degC"),  # 0.088; 0.088
    # Original: "Transfer coefficient for lower level"
    "c4": (0.00487, ""),  # 0.025; 0.025
    # Original: Radiative forcing due to CO2 doubling (Wm-2)
    "fco22x": (3.8, "W/m^2"),  # 3.6813
    # Original: "2010 forcings of non-CO2 GHG (Wm-2)"
    "fex0": (0.25, "W/m^2"),  # 0.5
    # Original: "2100 forcings of non-CO2 GHG (Wm-2)"
    "fex1": (0.7, "W/m^2"),  # 1.0
    # Period length in seconds (not part of original)
    "period_length": (YEAR, None),
    # Use original conversion factor from tCO2 to tC
    # (has rounding errors but needed to reproduce original output; not part of original)
    "original_rounding": (True, None),
    # Time when forcing due to other greenhouse gases saturates (original: 2100)
    "forcoth_saturation_time": (np.datetime64("2100-01-01"), None),
    # Equilibrium climate sensitivity
    #     Original: "Equilibrium temp impact (oC per doubling CO2)"
    "t2xco2": (2.9, "delta_degC"),  # 3.1
    # Not in original
    "start_time": (_default_start_time, None),
    "stop_time": (_default_stop_time, None),
    "E": (None, "GtCO2/a", "average"),
}


class DICE(Adapter):
    """
    Adapter for the climate component from the Dynamic Integrated Climate-Economy (DICE)
    model.

    TODO: use original calibration

    TODO: look at DICE original documentation to work out what it's convention for emissions
    and radiative forcing is. It could actually be point, I need to check (sorry Sven for all 
    this mucking around, one day it will end...)
    """

    _timestep: int
    """Current time step"""

    _timestep_count: int
    """Total number of time steps"""

    _values: Any
    """Parameter views"""

    _openscm_standard_parameter_mappings = {
        "Equilibrium Climate Sensitivity": "t2xco2",
        "Radiative Forcing 2xCO2": "fco22x",
        "Start Time": "start_time",
        "Stop Time": "stop_time",
        "Step Length": "period_length",
        ("Emissions", "CO2"): "E",
    }

    @property
    def name(self):
        return "DICE"

    def _initialize_model(self) -> None:
        """
        Initialize the model.
        """
        parameter_names = list(MODEL_PARAMETER_DEFAULTS.keys()) + [
            "mat",
            "ml",
            "mu",
            "tatm",
            "tocean",
            "forc",
            "b11",
            "b21",
            "b22",
            "b32",
            "b33",
        ]
        self._values = namedtuple("DICEViews", parameter_names)

        imap = self._inverse_openscm_standard_parameter_mappings
        for name, settings in MODEL_PARAMETER_DEFAULTS.items():
            full_name = ("DICE", name)
            if len(settings) == 2:
                default, unit = settings

                self._add_parameter_view(full_name, value=default, unit=unit)
                setattr(self._values, name, self._parameter_views[full_name])

                if name in imap:
                    openscm_name = imap[name]
                    # don't set default here, leave that for later
                    self._add_parameter_view(openscm_name, unit=unit)

            else:
                default, unit, timeseries_type = settings
                self._add_parameter_view(
                    full_name, value=default, unit=unit, timeseries_type=timeseries_type
                )
                setattr(self._values, name, self._parameter_views[full_name])

                if name in imap:
                    openscm_name = imap[name]
                    # don't set default here, leave that for later
                    self._add_parameter_view(
                        openscm_name, unit=unit, timeseries_type=timeseries_type
                    )

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
        except:
            return self._parameter_views[("DICE", "start_time")].value

    @property
    def _period_length(self):
        try:
            return self._parameter_views["Step Length"].value
        except:
            return self._parameter_views[("DICE", "period_length")].value

    @property
    def _timestep_count(self):
        try:
            stop_time = self._parameter_views["Stop Time"].value
        except:
            stop_time = self._parameter_views[("DICE", "stop_time")].value

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
                model_n = (self.name, self._openscm_standard_parameter_mappings[n])
                if (
                    self._parameter_views[model_n].version
                    > self._parameter_versions[model_n]
                ):
                    return True
        return False

    def _update_model(self, name: HierarchicalName, para: ParameterInfo) -> None:
        try:
            values = self._get_parameter_value(para)
            if name in self._openscm_standard_parameter_mappings:
                model_name = (
                    self.name,
                    self._openscm_standard_parameter_mappings[name],
                )
                self._check_derived_paras([model_name], name)
                setattr(self._values, model_name[1], para)
                self._set_parameter_value(self._parameter_views[model_name], values)
            else:
                assert name[0] == self.name, "..."
                setattr(self._values, name[1], para)

        except ParameterEmptyError:
            pass

    def _reset(self) -> None:
        self._set_output_views()
        self._timestep = 0
        v = self._values  # just for convenience

        # Original: "Carbon cycle transition matrix"
        v.b11 = 1 - v.b12.value
        v.b21 = v.b12.value * v.mateq.value / v.mueq.value
        v.b22 = 1 - v.b21 - v.b23.value
        v.b32 = v.b23.value * v.mueq.value / v.mleq.value
        v.b33 = 1 - v.b32

        v.mat.values = np.empty(self._timestep_count)
        v.mat.values[0] = v.mat0.value
        v.ml.values = np.empty(self._timestep_count)
        v.ml.values[0] = v.ml0.value
        v.mu.values = np.empty(self._timestep_count)
        v.mu.values[0] = v.mu0.value
        v.tatm.values = np.empty(self._timestep_count)
        v.tatm.values[0] = v.tatm0.value
        v.tocean.values = np.empty(self._timestep_count)
        v.tocean.values[0] = v.tocean0.value

        v.forc.values = np.empty(self._timestep_count)
        v.forc.values[0] = (
            v.fco22x.value * log2(v.mat0.value / v.mateq.value) + v.fex0.value
        )

    def _set_output_views(self) -> None:
        # Original: "Total CO2 emissions (GtCO2 per year)"
        self._values.E = self._parameters.timeseries(
            ("Emissions", "CO2"),
            "GtCO2/a" if self._values.original_rounding.value else "GtC/a",
            self._get_time_points("average"),
            timeseries_type="average",
        )

        # Original: "Carbon concentration increase in atmosphere (GtC from 1750)"
        self._values.mat = self._output.timeseries(
            ("Pool", "CO2", "Atmosphere"),
            "GtC",
            self._get_time_points("point"),
            timeseries_type="point",
        )

        # Original: "Carbon concentration increase in lower oceans (GtC from 1750)"
        self._values.ml = self._output.timeseries(
            ("Pool", "CO2", "Ocean", "lower"),
            "GtC",
            self._get_time_points("point"),
            timeseries_type="point",
        )

        # Original: "Carbon concentration increase in shallow oceans (GtC from 1750)"
        self._values.mu = self._output.timeseries(
            ("Pool", "CO2", "Ocean", "shallow"),
            "GtC",
            self._get_time_points("point"),
            timeseries_type="point",
        )

        # Original: "Increase temperature of atmosphere (degrees C from 1900)"
        self._values.tatm = self._output.timeseries(
            ("Surface Temperature Increase"),
            "delta_degC",
            self._get_time_points("point"),
            timeseries_type="point",
        )

        # Original: "Increase in temperatureof lower oceans (degrees from 1900)"
        self._values.tocean = self._output.timeseries(
            ("Ocean Temperature Increase"),
            "delta_degC",
            self._get_time_points("point"),
            timeseries_type="point",
        )

        # Original: "Increase in radiative forcing (watts per m2 from 1900)"
        self._values.forc = self._output.timeseries(
            ("Radiative Forcing", "CO2"),
            "W/m^2",
            self._get_time_points("average"),
            timeseries_type="average",
        )

    def _shutdown(self) -> None:
        pass

    def _run(self) -> None:
        """
        Run the model over the full time range.
        """
        v = self._values  # just for convenience

        v.mat.lock()
        v.ml.lock()
        v.mu.lock()
        v.tatm.lock()
        v.tocean.lock()
        v.forc.lock()

        for _ in range(self._timestep_count - 1):
            self._calc_step()

        v.mat.unlock()
        v.ml.unlock()
        v.mu.unlock()
        v.tatm.unlock()
        v.tocean.unlock()
        v.forc.unlock()

    def _step(self) -> None:
        """
        Do a single time step.
        """
        self._calc_step()

    def _calc_step(self) -> None:
        """
        Calculate a single time step.
        """
        self._timestep += 1
        self._current_time += self._values.period_length.value
        v = self._values  # just for convenience

        # Original: "Carbon concentration increase in atmosphere (GtC from 1750)"
        v.mat.values[self._timestep] = max(
            v.mat_lower.value,
            v.mat.values[self._timestep - 1] * v.b11
            + v.mu.values[self._timestep - 1] * v.b21
            + v.E.values[self._timestep - 1]
            * float(v.period_length.value / YEAR)
            / (3.666 if v.original_rounding.value else 1),
        )

        # Original: "Carbon concentration increase in lower oceans (GtC from 1750)"
        v.ml.values[self._timestep] = max(
            v.ml_lower.value,
            v.ml.values[self._timestep - 1] * v.b33
            + v.mu.values[self._timestep - 1] * v.b23.value,
        )

        # Original: "Carbon concentration increase in shallow oceans (GtC from 1750)"
        v.mu.values[self._timestep] = max(
            v.mu_lower.value,
            v.mat.values[self._timestep - 1] * v.b12.value
            + v.mu.values[self._timestep - 1] * v.b22
            + v.ml.values[self._timestep - 1] * v.b32,
        )

        # Original: "Increase temperatureof lower oceans (degrees C from 1900)" (sic)
        v.tocean.values[self._timestep] = max(
            v.tocean_lower.value,
            min(
                v.tocean_upper.value,
                v.tocean.values[self._timestep - 1]
                + v.c4.value
                * (
                    v.tatm.values[self._timestep - 1]
                    - v.tocean.values[self._timestep - 1]
                ),
            ),
        )

        # Original: "Exogenous forcing for other greenhouse gases"
        if (
            v.start_time.value + v.period_length.value * self._timestep
            >= v.forcoth_saturation_time.value
        ):
            forcoth = v.fex1.value
        else:
            forcoth = v.fex0.value + (v.fex1.value - v.fex0.value) * (
                v.period_length.value * self._timestep
            ) / (v.forcoth_saturation_time.value - v.start_time.value)

        # Original: "Increase in radiative forcing (watts per m2 from 1900)"
        # import pdb
        # pdb.set_trace()
        v.forc.values[self._timestep] = (
            v.fco22x.value * log2(v.mat.values[self._timestep] / v.mateq.value)
            + forcoth
        )

        # Original: "Increase temperature of atmosphere (degrees C from 1900)"
        v.tatm.values[self._timestep] = min(
            v.tatm_upper.value,
            v.tatm.values[self._timestep - 1]
            + v.c1.value
            * (
                v.forc.values[self._timestep]
                - (v.fco22x.value / v.t2xco2.value) * v.tatm.values[self._timestep - 1]
                - v.c3.value
                * (
                    v.tatm.values[self._timestep - 1]
                    - v.tocean.values[self._timestep - 1]
                )
            ),
        )
