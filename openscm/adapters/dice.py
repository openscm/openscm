from ..adapter import Adapter


class DICE(Adapter):
    """
    TODO
    """

    _b12 = 0.0181
    """Carbon cycle transition matrix"""

    _b23 = 0.00071
    """Carbon cycle transition matrix"""

    _M_atm0 = 830.4
    """Initial Concentration in atmosphere 2010 (GtC)"""

    _M_atm_lower = 10

    _M_atm_eq = 588
    """Equilibrium concentration atmosphere (GtC)"""

    _M_u0 = 1527
    """Initial Concentration in upper strata 2010 (GtC)"""

    _M_u_lower = 100

    _M_u_eq = 1350
    """Equilibrium concentration in upper strata (GtC)"""

    _M_l0 = 10010
    """Initial Concentration in lower strata 2010 (GtC)"""

    _M_l_lower = 1000

    _M_l_eq = 10000
    """Equilibrium concentration in lower strata (GtC)"""

    _t2xco2 = 2.9
    """Equilibrium temp impact (oC per doubling CO2)"""

    _fex0 = 0.25
    """2010 forcings of non-CO2 GHG (Wm-2)"""

    _fex1 = 0.7
    """2100 forcings of non-CO2 GHG (Wm-2)"""

    _T_ocean0 = 0.0068
    """Initial lower stratum temp change (C from 1900)"""

    _T_ocean_lower = -1

    _T_ocean_upper = 20

    _T_atm0 = 0.8
    """Initial atmospheric temp change (C from 1900)"""

    _T_atm_upper = 40

    _c1 = 0.0222
    """Climate equation coefficient for upper level"""

    _c3 = 0.09175
    """Transfer coefficient upper to lower stratum"""

    _c4 = 0.00487
    """Transfer coefficient for lower level"""

    _fco22x = 3.8
    """Forcings of equilibrium CO2 doubling (Wm-2)"""

    def __init__(self, parameters: ParameterSet):
        super(self).__init__(parameters)

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model.
        """

        # TODO: Read parameters

        # Carbon cycle transition matrix:
        self._b11 = 1 - self._b12
        self._b21 = self._b12 * self._M_atm_eq / self._M_u_eq
        self._b22 = 1 - self._b21 - self._b23
        self._b32 = self._b23 * self._M_u_eq / self._M_l_eq
        self._b33 = 1 - self._b32

        period_length = 365 * 24 * 3600
        # TODO: start_time

        # Total CO2 emissions (GtCO2 per year)
        self._E = self._parameterset.get_writable_timeseries_view(
            ("Emissions", "CO2"), (), "GtCO2/a", start_time, period_length
        )

        # TODO:
        # Concentration in atmosphere 2010 (GtC)
        self._M_atm = self._parameterset.get_writable_timeseries_view(
            ("",), (), "GtC", start_time, period_length
        )
        self._M_atm.set(0, self._M_atm0)

        # Carbon concentration increase in lower oceans (GtC from 1750)
        self._M_l = self._parameterset.get_writable_timeseries_view(
            ("",), (), "GtC", start_time, period_length
        )
        self._M_l.set(0, self._M_l0)

        # Carbon concentration increase in shallow oceans (GtC from 1750)
        self._M_u = self._parameterset.get_writable_timeseries_view(
            ("",), (), "GtC", start_time, period_length
        )
        self._M_u.set(0, self._M_u0)

        # Increase in temperature of lower oceans (degrees C from 1900)
        self._T_ocean = self._parameterset.get_writable_timeseries_view(
            ("",), (), "°C", start_time, period_length
        )
        self._T_ocean.set(0, self._T_ocean0)

        # Exogenous forcing for other greenhouse gases
        self._forcoth = self._parameterset.get_writable_timeseries_view(
            ("",), (), "W/m^2", start_time, period_length
        )
        # TODO:
        # Constant forcoth(Time t) {
        #     const Time year = global.start_year + global.timestep_length * t;
        # if (year > 2100) {
        #     return fex1;
        # } else {
        #     return fex0 + (fex1 - fex0) * (global.timestep_length * 0.2 * t) / 18;
        # }
        # }

        # Increase in radiative forcing (watts per m2 from 1900)
        self._force = self._parameterset.get_writable_timeseries_view(
            ("",), (), "W/m^2", start_time, period_length
        )
        # TODO: set t=0 value

        # Increase temperature of atmosphere (degrees C from 1900)
        self._T_atm = self._parameterset.get_writable_timeseries_view(
            ("",), (), "°C", start_time, period_length
        )

    @abstractmethod
    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        """
        Do a single time step.
        """
        # TODO: t =
        # TODO: assert t > 0

        """
        TODO:
        # Concentration in atmosphere 2010 (GtC)
        M_atm.set(t, max(M_atm_lower, M_atm_last * b11 + M_u(t - 1) * b21 + E(t - 1) * global.timestep_length / 3.666))

        # Carbon concentration increase in lower oceans (GtC from 1750)
        M_l.set(t, max(M_l_lower, M_l_last * b33 + M_u(t - 1) * b23))

        # Carbon concentration increase in shallow oceans (GtC from 1750)
        M_u.set(t, max(M_u_lower, M_atm(t - 1) * b12 + M_u_last * b22 + M_l(t - 1) * b32))

        # Increase in temperature of lower oceans (degrees C from 1900)
        M_oceat.set(t, max(T_ocean_lower, min(T_ocean_upper, T_ocean_last + c4 * (T_atm(t - 1) - T_ocean_last))))

        # Increase in radiative forcing (watts per m2 from 1900)
        force.set(t, fco22x * std::log2(M_atm(t) / M_atm_eq) + forcoth(t))

        # Increase temperature of atmosphere (degrees C from 1900)
        T_atm.set(t, min(T_atm_upper, T_atm_last + c1 * (force(t) - (fco22x / t2xco2) * T_atm_last - c3 * (T_atm_last - T_ocean(t - 1)))))
        """
