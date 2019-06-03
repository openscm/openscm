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
from typing import Any

import numpy as np

from ..core.parameters import ParameterType
from ..core.time import create_time_points
from . import Adapter

YEAR = np.timedelta64(365, "D")  # pylint: disable=too-many-function-args

MODEL_PARAMETER_DEFAULTS = {
    # Initial size of atmospheric CO2 pool
    #     Original: "Initial Concentration in atmosphere 2010 (GtC)"
    "mat0": (830.4, "GtC"),  # 851
    # Equilibrium size of atmospheric CO2 pool
    #     Original: "Equilibrium concentration atmosphere (GtC)"
    "mateq": (588, "GtC"),  # 588
    # do we know what this is?
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
    # Radiative forcing due CO2 doubling (Wm-2)
    "fco22x": (3.8, "W/m^2"),  # 3.6813
    # Original: "2010 forcings of non-CO2 GHG (Wm-2)"
    "fex0": (0.25, "W/m^2"),  # 0.5
    # Original: "2100 forcings of non-CO2 GHG (Wm-2)"
    "fex1": (0.7, "W/m^2"),  # 1.0
    # Equilibrium climate sensitivity
    #     Original: "Equilibrium temp impact (oC per doubling CO2)"
    "t2xco2": (2.9, "delta_degC"),  # 3.1
    # Period length in seconds (not part of original)
    "period_length": (YEAR, None),
    # Use original conversion factor from tCO2 to tC
    # (has rounding errors but needed to reproduce original output; not part of original)
    "original_rounding": (True, None),
    # Time when forcing due to other greenhouse gases saturates (original: 2100)
    "forcoth_saturation_time": (np.datetime64("2100-01-01"), None),
}


class DICE(Adapter):
    """
    Adapter for the climate component from the Dynamic Integrated Climate-Economy (DICE)
    model.

    TODO: use original calibration
    """

    _timestep: int
    """Current time step"""

    _timestep_count: int
    """Total number of time steps"""

    _values: Any
    """Parameter views"""

    def _initialize_model(self) -> None:
        """
        Initialize the model.
        """
        parameter_names = list(MODEL_PARAMETER_DEFAULTS.keys()) + [
            "E",
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

        for name, (default, unit) in MODEL_PARAMETER_DEFAULTS.items():
            if unit is None:
                # Non-scalar parameter
                self._parameters.generic(("DICE", name)).value = default
                setattr(self._values, name, self._parameters.generic(("DICE", name)))
            else:
                # Scalar parameter
                self._parameters.scalar(("DICE", name), unit).value = default
                setattr(
                    self._values, name, self._parameters.scalar(("DICE", name), unit)
                )

    def _initialize_model_input(self) -> None:
        pass

    def _initialize_run_parameters(self) -> None:
        self._timestep = 0
        self._timestep_count = (
            int((self._stop_time - self._start_time) / self._values.period_length.value)
            + 1
        )  # include self._stop_time

        time_points = create_time_points(
            self._start_time,
            self._values.period_length.value,
            self._timestep_count,
            ParameterType.POINT_TIMESERIES,
        )
        time_points_for_averages = create_time_points(
            self._start_time,
            self._values.period_length.value,
            self._timestep_count,
            ParameterType.AVERAGE_TIMESERIES,
        )

        # Original: "Total CO2 emissions (GtCO2 per year)"
        self._values.E = self._parameters.timeseries(
            ("Emissions", "CO2"),
            "GtCO2/a" if self._values.original_rounding.value else "GtC/a",
            time_points_for_averages,
            timeseries_type="average",
        )

        # Original: "Carbon concentration increase in atmosphere (GtC from 1750)"
        self._values.mat = self._output.timeseries(
            ("Pool", "CO2", "Atmosphere"), "GtC", time_points, timeseries_type="point"
        )

        # Original: "Carbon concentration increase in lower oceans (GtC from 1750)"
        self._values.ml = self._output.timeseries(
            ("Pool", "CO2", "Ocean", "lower"),
            "GtC",
            time_points,
            timeseries_type="point",
        )

        # Original: "Carbon concentration increase in shallow oceans (GtC from 1750)"
        self._values.mu = self._output.timeseries(
            ("Pool", "CO2", "Ocean", "shallow"),
            "GtC",
            time_points,
            timeseries_type="point",
        )

        # Original: "Increase temperature of atmosphere (degrees C from 1900)"
        self._values.tatm = self._output.timeseries(
            ("Surface Temperature", "Increase"),
            "delta_degC",
            time_points,
            timeseries_type="point",
        )

        # Original: "Increase in temperatureof lower oceans (degrees from 1900)"
        self._values.tocean = self._output.timeseries(
            ("Ocean Temperature", "Increase"),
            "delta_degC",
            time_points,
            timeseries_type="point",
        )

        # Original: "Increase in radiative forcing (watts per m2 from 1900)"
        self._values.forc = self._output.timeseries(
            ("Radiative Forcing", "CO2"), "W/m^2", time_points, timeseries_type="point"
        )  # TODO: convert to average (seems surprising RF would be being used as point..)?

    def _reset(self) -> None:
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

    def _shutdown(self) -> None:
        pass

    def _run(self) -> None:
        """
        Run the model over the full time range.
        """
        for _ in range(
            self._timestep_count - 1
        ):  # TODO: add lock mechanism for parameter views for performance
            self._calc_step()

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
            self._start_time + v.period_length.value * self._timestep
            >= v.forcoth_saturation_time.value
        ):
            forcoth = v.fex1.value
        else:
            forcoth = v.fex0.value + (v.fex1.value - v.fex0.value) * (
                v.period_length.value * self._timestep
            ) / (v.forcoth_saturation_time.value - self._start_time)

        # Original: "Increase in radiative forcing (watts per m2 from 1900)"
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
