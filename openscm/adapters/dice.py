from .. import constants
from ..adapter import Adapter
from datetime import datetime
from math import log2


class DICE(Adapter):
    """
    Climate component from the Dynamic Integrated Climate-Economy (DICE) model by
    William Nordhaus

    https://sites.google.com/site/williamdnordhaus/dice-rice

    TODO recalculate increase to absolute values
    TODO set force[0]
    TODO change to fluxes
    TODO add array handling reading and writing timeseries
    """

    _M_atm0: float
    """Initial concentration in atmosphere 2010 (GtC)"""

    _M_atm_eq: float
    """Equilibrium concentration atmosphere (GtC)"""

    _M_atm_lower: float
    """Lower bound concentration in atmosphere 2010 (GtC)"""

    _M_l0: float
    """Initial concentration in lower strata 2010 (GtC)"""

    _M_l_eq: float
    """Equilibrium concentration in lower strata (GtC)"""

    _M_l_lower: float
    """Lower bound for concentration in lower strata 2010 (GtC)"""

    _M_u0: float
    """Initial concentration in upper strata 2010 (GtC)"""

    _M_u_eq: float
    """Equilibrium concentration in upper strata (GtC)"""

    _M_u_lower: float
    """Lower bound for concentration in upper strata 2010 (GtC)"""

    _T_atm0: float
    """Initial atmospheric temp change (°C from 1900)"""

    _T_atm_upper: float
    """Upper bound for atmospheric temp change (°C from 1900)"""

    _T_ocean0: float
    """Initial lower stratum temp change (°C from 1900)"""

    _T_ocean_lower: float
    """Lower bound for lower stratum temp change (°C from 1900)"""

    _T_ocean_upper: float
    """Upper bound for lower stratum temp change (°C from 1900)"""

    _b12: float
    """Carbon cycle transition matrix"""

    _b23: float
    """Carbon cycle transition matrix"""

    _c1: float
    """Climate equation coefficient for upper level"""

    _c3: float
    """Transfer coefficient upper to lower stratum"""

    _c4: float
    """Transfer coefficient for lower level"""

    _fex0: float
    """2010 forcings of non-CO2 GHG (Wm-2)"""

    _fex1: float
    """2100 forcings of non-CO2 GHG (Wm-2)"""

    _fco22x: float
    """Forcings of equilibrium CO2 doubling (Wm-2)"""

    _period_length: int
    """Period length in seconds (currently 1yr)"""

    _t2xco2: float
    """Equilibrium temp impact (°C per doubling CO2)"""

    _timestep: int
    """Current time step"""

    _timestep_count: int
    """Total number of time steps"""

    _year2100: int
    """Timestamp of 2100-01-01"""

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model.
        """

        self._period_length = constants.YEAR
        self._timestep = 0
        self._timestep_count = (
            self._stop_time - self._start_time
        ) // self._period_length
        self._year2100 = datetime(2100, 1, 1).timestamp()

        # Read model parameters
        model_parameter_defaults = {
            # Initial concentration in atmosphere 2010 (GtC)
            "M_atm0": (830.4, "GtC/a"),
            # Equilibrium concentration atmosphere (GtC)
            "M_atm_eq": (588, "GtC/a"),
            # Lower bound concentration in atmosphere 2010 (GtC)
            "M_atm_lower": (10, "GtC/a"),
            # Initial concentration in lower strata 2010 (GtC)
            "M_l0": (10010, "GtC/a"),
            # Equilibrium concentration in lower strata (GtC)
            "M_l_eq": (10000, "GtC/a"),
            # Lower bound for concentration in lower strata 2010 (GtC)
            "M_l_lower": (1000, "GtC/a"),
            # Initial concentration in upper strata 2010 (GtC)
            "M_u0": (1527, "GtC/a"),
            # Equilibrium concentration in upper strata (GtC)
            "M_u_eq": (1350, "GtC/a"),
            # Lower bound for concentration in upper strata 2010 (GtC)
            "M_u_lower": (100, "GtC/a"),
            # Initial atmospheric temp change (°C from 1900)
            "T_atm0": (0.8, "°C"),
            # Upper bound for atmospheric temp change (°C from 1900)
            "T_atm_upper": (40, "°C"),
            # Initial lower stratum temp change (°C from 1900)
            "T_ocean0": (0.0068, "°C"),
            # Lower bound for lower stratum temp change (°C from 1900)
            "T_ocean_lower": (-1, "°C"),
            # Upper bound for lower stratum temp change (°C from 1900)
            "T_ocean_upper": (20, "°C"),
            # Carbon cycle transition matrix
            "b12": (0.0181, ""),
            # Carbon cycle transition matrix
            "b23": (0.00071, ""),
            # Climate equation coefficient for upper level
            "c1": (0.0222, "°C*m^2/W"),
            # Transfer coefficient upper to lower stratum
            "c3": (0.09175, "W/m^2/°C"),
            # Transfer coefficient for lower level
            "c4": (0.00487, ""),
            # Forcings of equilibrium CO2 doubling (Wm-2)
            "fco22x": (3.8, "W/m^2"),
            # 2010 forcings of non-CO2 GHG (Wm-2)
            "fex0": (0.25, "W/m^2"),
            # 2100 forcings of non-CO2 GHG (Wm-2)
            "fex1": (0.7, "W/m^2"),
            # Equilibrium temp impact (°C per doubling CO2)
            "t2xco2": (2.9, "°C"),
        }

        for n, v in model_parameter_defaults.items():
            default, unit = v
            setattr(
                self,
                "_{}".format(n),
                self._parameterset.get_scalar_view(("DICE", n), (), unit).get(default),
            )

        # Carbon cycle transition matrix
        self._b11 = 1 - self._b12
        self._b21 = self._b12 * self._M_atm_eq / self._M_u_eq
        self._b22 = 1 - self._b21 - self._b23
        self._b32 = self._b23 * self._M_u_eq / self._M_l_eq
        self._b33 = 1 - self._b32

        # Total CO2 emissions (GtCO2 per year)
        self._E = self._parameterset.get_writable_timeseries_view(
            ("Emissions", "CO2"), (), "GtCO2/a", self._start_time, self._period_length
        )

        # Concentration in atmosphere (GtC)
        self._M_atm = self._parameterset.get_writable_timeseries_view(
            ("Concentration", "Atmosphere"),
            (),
            "GtC",
            self._start_time,
            self._period_length,
        )
        self._M_atm.set(0, self._M_atm0)

        # Carbon concentration increase in lower oceans (GtC from 1750)
        self._M_l = self._parameterset.get_writable_timeseries_view(
            ("Concentration", "Ocean", "lower"),
            (),
            "GtC",
            self._start_time,
            self._period_length,
        )
        self._M_l.set(0, self._M_l0)

        # Carbon concentration increase in shallow oceans (GtC from 1750)
        self._M_u = self._parameterset.get_writable_timeseries_view(
            ("Concentration", "Ocean", "shallow"),
            (),
            "GtC",
            self._start_time,
            self._period_length,
        )
        self._M_u.set(0, self._M_u0)

        # Increase in temperature of lower oceans (°C from 1900)
        self._T_ocean = self._parameterset.get_writable_timeseries_view(
            ("Temperature Increase", "Ocean", "lower"),
            (),
            "°C",
            self._start_time,
            self._period_length,
        )
        self._T_ocean.set(0, self._T_ocean0)

        # Increase in radiative forcing (watts per m2 from 1900)
        self._force = self._parameterset.get_writable_timeseries_view(
            ("Radiative forcing",), (), "W/m^2", self._start_time, self._period_length
        )

        # Increase temperature of atmosphere (°C from 1900)
        self._T_atm = self._parameterset.get_writable_timeseries_view(
            ("Temperature Increase", "Atmosphere"),
            (),
            "°C",
            self._start_time,
            self._period_length,
        )

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        self._timestep = 0
        for _ in range(self._timestep_count):
            self.step()

    def step(self) -> None:
        """
        Do a single time step.
        """

        self._timestep += 1

        # Concentration in atmosphere (GtC)
        this_M_atm = max(
            self._M_atm_lower,
            last_M_atm * self._b11
            + last_M_u * self._b21
            + E[self._timestep - 1] * self._period_length / constants.YEAR / 3.666,
        )

        # Carbon concentration increase in lower oceans (GtC from 1750)
        this_M_l = max(self._M_l_lower, last_M_l * self._b33 + last_M_u * self._b23)

        # Carbon concentration increase in shallow oceans (GtC from 1750)
        this_M_u = max(
            self._M_u_lower,
            last_M_atm * self._b12 + last_M_u * self._b22 + last_M_l * self._b32,
        )

        # Increase in temperature of lower oceans (°C from 1900)
        this_T_ocean = max(
            self._T_ocean_lower,
            min(
                self._T_ocean_upper,
                last_T_ocean + self._c4 * (last_T_atm - last_T_ocean),
            ),
        )

        # Exogenous forcing for other greenhouse gases
        if self._start_time + self._period_length * self._timestep > self._year2100:
            forcoth = self._fex1
        else:
            forcoth = (
                self._fex0
                + (self._fex1 - self._fex0)
                * (self._period_length / constants.YEAR * 0.2 * self._timestep)
                / 18
            )

        # Increase in radiative forcing (watts per m2 from 1900)
        this_force = self._fco22x * log2(this_M_atm / self._M_atm_eq) + forcoth

        # Increase temperature of atmosphere (°C from 1900)
        this_T_atm = min(
            self._T_atm_upper,
            self._T_atm_last
            + self._c1
            * (
                this_force
                - (self._fco22x / self._t2xco2) * last_T_atm
                - self._c3 * (last_T_atm - lst_T_ocean)
            ),
        )
