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
from datetime import datetime
from math import log2
from typing import Any

import numpy as np

from ..core.parameters import ParameterType
from ..core.time import ExtrapolationType, InterpolationType, create_time_points
from . import Adapter

YEAR = 365 * 24 * 60 * 60

MODEL_PARAMETER_DEFAULTS = {
    # Initial pool size atmosphere
    #     Original: "Initial Concentration in atmosphere 2010 (GtC)"
    "mat0": (830.4, "GtC"),  # 851
    # Equilibrium pool size atmosphere
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
    "tatm0": (0.8, "degC"),  # 0.85
    "tatm_upper": (40, "degC"),  # 12
    # Original: "Initial lower stratum temp change (C from 1900)"
    "tocean0": (0.0068, "degC"),  # 0.0068
    "tocean_lower": (-1, "degC"),  # -1
    "tocean_upper": (20, "degC"),  # 20
    # Original: "Carbon cycle transition matrix"
    "b12": (0.0181, ""),  # 0.088; 0.12
    # Original: "Carbon cycle transition matrix"
    "b23": (0.00071, ""),  # 0.00250; 0.007
    # Original: "Climate equation coefficient for upper level"
    "c1": (0.0222, "degC*m^2/W"),  # 0.098; 0.1005
    # Original: "Transfer coefficient upper to lower stratum"
    "c3": (0.09175, "W/m^2/degC"),  # 0.088; 0.088
    # Original: "Transfer coefficient for lower level"
    "c4": (0.00487, ""),  # 0.025; 0.025
    # Forcings of equilibrium CO2 doubling (Wm-2)
    "fco22x": (3.8, "W/m^2"),  # 3.6813
    # Original: "2010 forcings of non-CO2 GHG (Wm-2)"
    "fex0": (0.25, "W/m^2"),  # 0.5
    # Original: "2100 forcings of non-CO2 GHG (Wm-2)"
    "fex1": (0.7, "W/m^2"),  # 1.0
    # Equilibrium climate sensitivity
    #     Original: "Equilibrium temp impact (oC per doubling CO2)"
    "t2xco2": (2.9, "degC"),  # 3.1
    # Period length in seconds (not part of original)
    "period_length": (YEAR, None),
    # Use original conversion factor from tCO2 to tC
    # (has rounding errors but needed to reproduce original output; not part of original)
    "original_rounding": (True, None),
    # Time when forcing due to other greenhouse gases saturates (original: 2100)
    "forcoth_saturation_time": (int(datetime(2100, 1, 1).timestamp()), None),
}


class DICE(Adapter):
    """
    Adapter for the climate component from the Dynamic Integrated Climate-Economy (DICE)
    model.

    TODO Use original calibration
    """

    _timestep: int
    """Current time step"""

    _timestep_count: int
    """Total number of time steps"""

    _values: Any
    """Parameter values"""

    _views: Any
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
        self._values = namedtuple("DICEValues", parameter_names)
        self._views = namedtuple("DICEViews", parameter_names)

        for name, (default, unit) in MODEL_PARAMETER_DEFAULTS.items():
            setattr(self._values, name, default)
            if unit is None:
                # Non-scalar parameter
                self._parameters.generic(("DICE", name), writable=True).value = default
                setattr(self._views, name, self._parameters.generic(("DICE", name)))
            else:
                # Scalar parameter
                self._parameters.scalar(
                    ("DICE", name), unit, writable=True
                ).value = default
                setattr(
                    self._views, name, self._parameters.scalar(("DICE", name), unit)
                )

    def _initialize_model_input(self) -> None:
        pass

    def _initialize_run_parameters(self) -> None:
        self._timestep = 0
        self._timestep_count = (self._stop_time - self._start_time) // int(
            self._values.period_length
        ) + 1  # include self._stop_time

        time_points = create_time_points(
            self._start_time,
            self._values.period_length,
            self._timestep_count,
            ParameterType.POINT_TIMESERIES,
        )
        time_points_for_averages = create_time_points(
            self._start_time,
            self._values.period_length,
            self._timestep_count,
            ParameterType.AVERAGE_TIMESERIES,
        )

        # Original: "Total CO2 emissions (GtCO2 per year)"
        self._views.E = self._parameters.timeseries(
            ("Emissions", "CO2"),
            "GtCO2/a" if self._values.original_rounding else "GtC/a",
            time_points_for_averages,
            timeseries_type="average",
        )

        # Original: "Carbon concentration increase in atmosphere (GtC from 1750)"
        self._views.mat = self._output.timeseries(
            ("Pool", "CO2", "Atmosphere"),
            "GtC",
            time_points,
            timeseries_type="point",
            writable=True,
        )

        # Original: "Carbon concentration increase in lower oceans (GtC from 1750)"
        self._views.ml = self._output.timeseries(
            ("Pool", "CO2", "Ocean", "lower"),
            "GtC",
            time_points,
            timeseries_type="point",
            writable=True,
        )

        # Original: "Carbon concentration increase in shallow oceans (GtC from 1750)"
        self._views.mu = self._output.timeseries(
            ("Pool", "CO2", "Ocean", "shallow"),
            "GtC",
            time_points,
            timeseries_type="point",
            writable=True,
        )

        # Original: "Increase temperature of atmosphere (degrees C from 1900)"
        self._views.tatm = self._output.timeseries(
            ("Surface Temperature", "Increase"),
            "degC",
            time_points,
            timeseries_type="point",
            writable=True,
        )

        # Original: "Increase in temperatureof lower oceans (degrees from 1900)"
        self._views.tocean = self._output.timeseries(
            ("Ocean Temperature", "Increase"),
            "degC",
            time_points,
            timeseries_type="point",
            writable=True,
        )

        # Original: "Increase in radiative forcing (watts per m2 from 1900)"
        self._views.forc = self._output.timeseries(
            ("Radiative Forcing", "CO2"),
            "W/m^2",
            time_points,
            timeseries_type="point",
            writable=True,
        )

    def _reset(self) -> None:
        self._timestep = 0
        for name in MODEL_PARAMETER_DEFAULTS:
            setattr(self._values, name, getattr(self._views, name).value)

        # Original: "Carbon cycle transition matrix"
        self._values.b11 = 1 - self._values.b12
        self._values.b21 = self._values.b12 * self._values.mateq / self._values.mueq
        self._values.b22 = 1 - self._values.b21 - self._values.b23
        self._values.b32 = self._values.b23 * self._values.mueq / self._values.mleq
        self._values.b33 = 1 - self._values.b32

        self._values.E = self._views.E.values

        self._values.mat = np.empty(self._timestep_count)
        self._values.mat[0] = self._values.mat0
        self._values.ml = np.empty(self._timestep_count)
        self._values.ml[0] = self._values.ml0
        self._values.mu = np.empty(self._timestep_count)
        self._values.mu[0] = self._values.mu0
        self._values.tatm = np.empty(self._timestep_count)
        self._values.tatm[0] = self._values.tatm0
        self._values.tocean = np.empty(self._timestep_count)
        self._values.tocean[0] = self._values.tocean0

        self._values.forc = np.empty(self._timestep_count)
        self._values.forc[0] = (
            self._values.fco22x * log2(self._values.mat0 / self._values.mateq)
            + self._values.fex0
        )

    def _shutdown(self) -> None:
        pass

    def _run(self) -> None:
        """
        Run the model over the full time range.
        """
        for _ in range(self._timestep_count - 1):
            self._calc_step()
        self._update_output()

    def _step(self) -> None:
        """
        Do a single time step.
        """
        self._calc_step()
        self._update_output()

    def _update_output(self) -> None:
        """
        Set output data from values.
        """
        self._views.mat.values = self._values.mat
        self._views.ml.values = self._values.ml
        self._views.mu.values = self._values.mu
        self._views.tatm.values = self._values.tatm
        self._views.tocean.values = self._values.tocean
        self._views.forc.values = self._values.forc

    def _calc_step(self) -> None:
        """
        Calculate a single time step.
        """
        self._timestep += 1
        self._current_time += YEAR

        # Original: "Carbon concentration increase in atmosphere (GtC from 1750)"
        self._values.mat[self._timestep] = max(
            self._values.mat_lower,
            self._values.mat[self._timestep - 1] * self._values.b11
            + self._values.mu[self._timestep - 1] * self._values.b21
            + self._values.E[self._timestep - 1]
            * self._values.period_length
            / YEAR
            / (3.666 if self._values.original_rounding else 1),
        )

        # Original: "Carbon concentration increase in lower oceans (GtC from 1750)"
        self._values.ml[self._timestep] = max(
            self._values.ml_lower,
            self._values.ml[self._timestep - 1] * self._values.b33
            + self._values.mu[self._timestep - 1] * self._values.b23,
        )

        # Original: "Carbon concentration increase in shallow oceans (GtC from 1750)"
        self._values.mu[self._timestep] = max(
            self._values.mu_lower,
            self._values.mat[self._timestep - 1] * self._values.b12
            + self._values.mu[self._timestep - 1] * self._values.b22
            + self._values.ml[self._timestep - 1] * self._values.b32,
        )

        # Original: "Increase temperatureof lower oceans (degrees C from 1900)" (sic)
        self._values.tocean[self._timestep] = max(
            self._values.tocean_lower,
            min(
                self._values.tocean_upper,
                self._values.tocean[self._timestep - 1]
                + self._values.c4
                * (
                    self._values.tatm[self._timestep - 1]
                    - self._values.tocean[self._timestep - 1]
                ),
            ),
        )

        # Original: "Exogenous forcing for other greenhouse gases"
        if (
            self._start_time + self._values.period_length * self._timestep
            >= self._values.forcoth_saturation_time
        ):
            forcoth = self._values.fex1
        else:
            forcoth = self._values.fex0 + (self._values.fex1 - self._values.fex0) * (
                self._values.period_length * self._timestep
            ) / (self._values.forcoth_saturation_time - self._start_time)

        # Original: "Increase in radiative forcing (watts per m2 from 1900)"
        self._values.forc[self._timestep] = (
            self._values.fco22x
            * log2(self._values.mat[self._timestep] / self._values.mateq)
            + forcoth
        )

        # Original: "Increase temperature of atmosphere (degrees C from 1900)"
        self._values.tatm[self._timestep] = min(
            self._values.tatm_upper,
            self._values.tatm[self._timestep - 1]
            + self._values.c1
            * (
                self._values.forc[self._timestep]
                - (self._values.fco22x / self._values.t2xco2)
                * self._values.tatm[self._timestep - 1]
                - self._values.c3
                * (
                    self._values.tatm[self._timestep - 1]
                    - self._values.tocean[self._timestep - 1]
                )
            ),
        )
