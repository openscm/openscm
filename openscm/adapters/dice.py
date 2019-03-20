"""
Adapter for the climate component from the Dynamic Integrated Climate-Economy (DICE)
model by William Nordhaus.

Original source: https://sites.google.com/site/williamdnordhaus/dice-rice
"""

from collections import namedtuple
from datetime import datetime
from math import log2
from typing import Any

import numpy as np

from ..adapter import Adapter
from ..parameters import ParameterType
from ..timeseries_converter import (
    ExtrapolationType,
    InterpolationType,
    create_time_points,
)

YEAR = 365 * 24 * 60 * 60

MODEL_PARAMETER_DEFAULTS = {
    # Initial concentration in atmosphere
    "M_atm0": (830.4, "GtC"),
    # Equilibrium concentration atmosphere
    "M_atm_eq": (588, "GtC"),
    # Lower bound concentration in atmosphere
    "M_atm_lower": (10, "GtC"),
    # Initial concentration in lower strata
    "M_l0": (10010, "GtC"),
    # Equilibrium concentration in lower strata
    "M_l_eq": (10000, "GtC"),
    # Lower bound for concentration in lower strata
    "M_l_lower": (1000, "GtC"),
    # Initial concentration in upper strata
    "M_u0": (1527, "GtC"),
    # Equilibrium concentration in upper strata
    "M_u_eq": (1350, "GtC"),
    # Lower bound for concentration in upper strata
    "M_u_lower": (100, "GtC"),
    # Initial atmospheric temperature change (rel. to 1900)
    "T_atm0": (0.8, "degC"),
    # Upper bound for atmospheric temperature change (rel. to 1900)
    "T_atm_upper": (40, "degC"),
    # Initial lower stratum temperature change (rel. to 1900)
    "T_ocean0": (0.0068, "degC"),
    # Lower bound for lower stratum temperature change (rel. to 1900)
    "T_ocean_lower": (-1, "degC"),
    # Upper bound for lower stratum temperature change (rel. to 1900)
    "T_ocean_upper": (20, "degC"),
    # Carbon cycle transition matrix
    "b12": (0.0181, ""),
    # Carbon cycle transition matrix
    "b23": (0.00071, ""),
    # Climate equation coefficient for upper level
    "c1": (0.0222, "degC*m^2/W"),
    # Transfer coefficient upper to lower stratum
    "c3": (0.09175, "W/m^2/degC"),
    # Transfer coefficient for lower level
    "c4": (0.00487, ""),
    # Forcings of equilibrium CO2 doubling
    "fco22x": (3.8, "W/m^2"),
    # 2010 forcings of non-CO2 GHG
    "fex0": (0.25, "W/m^2"),
    # 2100 forcings of non-CO2 GHG
    "fex1": (0.7, "W/m^2"),
    # Equilibrium climate sensitivity
    "t2xco2": (2.9, "degC"),
}


class DICE(Adapter):
    """
    Adapter for the climate component from the Dynamic Integrated Climate-Economy (DICE)
    model.

    TODO recalculate increase to absolute values
    TODO still calibrated to initial year 2010
    TODO What about timeseries parameters when run times change?
    """

    _period_length: int
    """Period length in seconds (currently 1yr)"""

    _timestep: int
    """Current time step"""

    _timestep_count: int
    """Total number of time steps"""

    _values: Any
    """Parameter values"""

    _views: Any
    """Parameter views"""

    _year2100: int
    """Timestamp of 2100-01-01"""

    def _initialize_model(self) -> None:
        """
        Initialize the model.
        """
        self._period_length = YEAR
        self._year2100 = int(datetime(2100, 1, 1).timestamp())  # TODO 365 days-year

        parameter_names = list(MODEL_PARAMETER_DEFAULTS.keys()) + [
            "E",
            "M_atm",
            "M_l",
            "M_u",
            "T_atm",
            "T_ocean",
            "force",
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
            self._parameters.get_writable_scalar_view(
                ("DICE", name), ("World",), unit
            ).set(default)
            setattr(
                self._views,
                name,
                self._parameters.get_scalar_view(("DICE", name), ("World",), unit),
            )

    def _initialize_model_input(self) -> None:
        pass

    def _initialize_run_parameters(self) -> None:
        self._timestep = 0
        self._timestep_count = (
            self._stop_time - self._start_time
        ) // self._period_length

        time_points = create_time_points(
            self._start_time,
            self._period_length,
            self._timestep_count,
            ParameterType.POINT_TIMESERIES,
        )
        time_points_for_averages = create_time_points(
            self._start_time,
            self._period_length,
            self._timestep_count,
            ParameterType.AVERAGE_TIMESERIES,
        )

        # Total CO2 emissions (GtCO2 per year)
        self._views.E = self._parameters.get_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "GtCO2/a",
            time_points_for_averages,
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

        # TODO remove setting missing emissions once there is a method in tests for that
        if self._views.E.is_empty:
            self._parameters.get_writable_timeseries_view(
                ("Emissions", "CO2"),
                ("World",),
                "GtCO2/a",
                time_points_for_averages,
                ParameterType.AVERAGE_TIMESERIES,
                InterpolationType.LINEAR,
                ExtrapolationType.LINEAR,
            ).set(np.zeros(self._timestep_count))

        # Concentration in atmosphere
        self._views.M_atm = self._output.get_writable_timeseries_view(
            ("Concentration", "Atmosphere"),
            ("World",),
            "GtC",
            time_points,
            ParameterType.POINT_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

        # Carbon concentration increase in lower oceans (rel. to 1750)
        self._views.M_l = self._output.get_writable_timeseries_view(
            ("Concentration", "Ocean", "lower"),
            ("World",),
            "GtC",
            time_points,
            ParameterType.POINT_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

        # Carbon concentration increase in shallow oceans (rel. to 1750)
        self._views.M_u = self._output.get_writable_timeseries_view(
            ("Concentration", "Ocean", "shallow"),
            ("World",),
            "GtC",
            time_points,
            ParameterType.POINT_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

        # Increase temperature of atmosphere (rel. to 1900)
        self._views.T_atm = self._output.get_writable_timeseries_view(
            ("Temperature Increase", "Atmosphere"),
            ("World",),
            "degC",
            time_points,
            ParameterType.POINT_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

        # Increase in temperature of lower oceans (rel. to 1900)
        self._views.T_ocean = self._output.get_writable_timeseries_view(
            ("Temperature Increase", "Ocean", "lower"),
            ("World",),
            "degC",
            time_points,
            ParameterType.POINT_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

        # Increase in radiative forcing (rel. to 1900)
        self._views.force = self._output.get_writable_timeseries_view(
            ("Radiative forcing",),
            ("World",),
            "W/m^2",
            time_points,
            ParameterType.POINT_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

    def _reset(self) -> None:
        self._timestep = 0
        for name in MODEL_PARAMETER_DEFAULTS:
            setattr(self._values, name, getattr(self._views, name).get())

        # Carbon cycle transition matrix
        self._values.b11 = 1 - self._values.b12
        self._values.b21 = (
            self._values.b12 * self._values.M_atm_eq / self._values.M_u_eq
        )
        self._values.b22 = 1 - self._values.b21 - self._values.b23
        self._values.b32 = self._values.b23 * self._values.M_u_eq / self._values.M_l_eq
        self._values.b33 = 1 - self._values.b32

        self._values.E = self._views.E.get()

        self._values.M_atm = np.empty(self._timestep_count)
        self._values.M_atm[0] = self._values.M_atm0
        self._values.M_l = np.empty(self._timestep_count)
        self._values.M_l[0] = self._values.M_l0
        self._values.M_u = np.empty(self._timestep_count)
        self._values.M_u[0] = self._values.M_u0
        self._values.T_atm = np.empty(self._timestep_count)
        self._values.T_atm[0] = self._values.T_atm0
        self._values.T_ocean = np.empty(self._timestep_count)
        self._values.T_ocean[0] = self._values.T_ocean0

        self._values.force = np.empty(self._timestep_count)
        self._values.force[0] = (
            self._values.fco22x * log2(self._values.M_atm0 / self._values.M_atm_eq)
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
        self._views.M_atm.set(self._values.M_atm)
        self._views.M_l.set(self._values.M_l)
        self._views.M_u.set(self._values.M_u)
        self._views.T_atm.set(self._values.T_atm)
        self._views.T_ocean.set(self._values.T_ocean)
        self._views.force.set(self._values.force)

    def _calc_step(self) -> None:
        """
        Calculate a single time step.
        """
        self._timestep += 1
        self._current_time += YEAR

        # Atmospheric pool size
        self._values.M_atm[self._timestep] = max(
            self._values.M_atm_lower,
            self._values.M_atm[self._timestep - 1] * self._values.b11
            + self._values.M_u[self._timestep - 1] * self._values.b21
            + self._values.E[self._timestep - 1] * self._period_length / YEAR / 44 / 12,
        )

        # Lower ocean pool size (rel. to 1750)
        self._values.M_l[self._timestep] = max(
            self._values.M_l_lower,
            self._values.M_l[self._timestep - 1] * self._values.b33
            + self._values.M_u[self._timestep - 1] * self._values.b23,
        )

        # Shallow ocean pool size (rel. to 1750)
        self._values.M_u[self._timestep] = max(
            self._values.M_u_lower,
            self._values.M_atm[self._timestep - 1] * self._values.b12
            + self._values.M_u[self._timestep - 1] * self._values.b22
            + self._values.M_l[self._timestep - 1] * self._values.b32,
        )

        # Increase in temperature of lower oceans (rel. to 1900)
        self._values.T_ocean[self._timestep] = max(
            self._values.T_ocean_lower,
            min(
                self._values.T_ocean_upper,
                self._values.T_ocean[self._timestep - 1]
                + self._values.c4
                * (
                    self._values.T_atm[self._timestep - 1]
                    - self._values.T_ocean[self._timestep - 1]
                ),
            ),
        )

        # Exogenous forcing for other greenhouse gases
        if self._start_time + self._period_length * self._timestep > self._year2100:
            forcoth = self._values.fex1
        else:
            forcoth = (
                self._values.fex0
                + (self._values.fex1 - self._values.fex0)
                * (self._period_length / YEAR * self._timestep)
                / 90.0
            )

        # Increase in radiative forcing (rel. to 1900)
        self._values.force[self._timestep] = (
            self._values.fco22x
            * log2(self._values.M_atm[self._timestep] / self._values.M_atm_eq)
            + forcoth
        )

        # Increase temperature of atmosphere (rel. to 1900)
        self._values.T_atm[self._timestep] = min(
            self._values.T_atm_upper,
            self._values.T_atm[self._timestep - 1]
            + self._values.c1
            * (
                self._values.force[self._timestep]
                - (self._values.fco22x / self._values.t2xco2)
                * self._values.T_atm[self._timestep - 1]
                - self._values.c3
                * (
                    self._values.T_atm[self._timestep - 1]
                    - self._values.T_ocean[self._timestep - 1]
                )
            ),
        )
