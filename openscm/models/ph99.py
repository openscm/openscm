from copy import deepcopy


import numpy as np


from ..units import unit_registry
from ..errors import OutOfBoundsError, OverwriteError

"""
TODO: put this somewhere
Decisions as I write:
- A model should take in a time start and have a timeperiod attribute. This avoids:
    - requiring models to interpolate internally, that should be somewhere else in pre-processing.
    - having to worry about irregular timesteps
        - with Pint, a month is just 1/12 of a year so that would also be a regular timestep from a Pint point of view
        - if people want to convert back to human calendars later, they can do so but that should also be a pre/post-processing step.
"""


class PH99Model(object):
    """Simple climate model first presented in Petschel-Held Climatic Change 1999

    This one box model projects global-mean |CO2| concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of |CO2| alone.

    Conventions:

    - All fluxes are time period averages and are assumed to be constant throughout
      the time period

    - All state variables are start of time period values


    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    def __init__(self, time_start=0 * unit_registry("yr")):
        """Initialise an instance of PH99Model

        Parameters
        ----------
        time_start: :obj:`pint.Quantity`
            Start time of run. `self.time_current` is set to this value.
        """
        self.time_start = time_start
        self.time_current = time_start

    _yr = 1 * unit_registry("yr")
    """:obj:`pint.Quantity`: one year"""

    @property
    def timestep(self):
        """:obj:`pint.Quantity`: Size of timestep"""
        return self._timestep * self._timestep_units

    @timestep.setter
    def timestep(self, value):
        self._timestep = value.to(self._timestep_units).magnitude

    @property
    def time_start(self):
        """:obj:`pint.Quantity`: Size of timestep"""
        return self._time_start * self._timestep_units

    @time_start.setter
    def time_start(self, value):
        self._time_start = value.to(self._timestep_units).magnitude

    @property
    def time_current(self):
        """:obj:`pint.Quantity`: Size of timestep"""
        return self._time_current * self._timestep_units

    @time_current.setter
    def time_current(self, value):
        self._time_current = value.to(self._timestep_units).magnitude

    _timestep_units = unit_registry("yr")
    _timestep = _yr.to(_timestep_units).magnitude

    @property
    def emissions(self):
        """`pint.Quantity` array: Emissions of |CO2|"""
        return self._emissions * self._emissions_units

    @emissions.setter
    def emissions(self, value):
        self._emissions = value.to(self._emissions_units).magnitude
        self._emissions_nan = np.isnan(np.sum(self._emissions))

    _emissions_units = unit_registry("GtC / yr")
    _emissions = np.array([np.nan])
    _emissions_nan = True

    @property
    def cumulative_emissions(self):
        """`pint.Quantity` array: Cumulative emissions of |CO2|"""
        return self._cumulative_emissions * self._cumulative_emissions_units

    @cumulative_emissions.setter
    def cumulative_emissions(self, value):
        self._cumulative_emissions = value.to(self._cumulative_emissions_units).magnitude

    _cumulative_emissions_units = unit_registry("GtC")
    _cumulative_emissions = np.array([np.nan])

    @property
    def concentrations(self):
        """`pint.Quantity` array: Concentrations of |CO2|"""
        return self._concentrations * self._concentrations_units

    @concentrations.setter
    def concentrations(self, value):
        self._concentrations = value.to(self._concentrations_units).magnitude

    _concentrations_units = unit_registry("ppm")
    _concentrations = np.array([np.nan])

    @property
    def temperatures(self):
        """`pint.Quantity` array: Global-mean temperatures"""
        return unit_registry.Quantity(self._temperatures, str(self._temperatures_units))

    @temperatures.setter
    def temperatures(self, value):
        self._temperatures = value.to(self._temperatures_units).magnitude

    # have to initialise like this to avoid ambiguity...
    _temperatures_tmp = unit_registry.Quantity(np.array([np.nan]), "degC")
    _temperatures_units = _temperatures_tmp.units
    _temperatures = _temperatures_tmp.magnitude

    @property
    def b(self):
        """:obj:`pint.Quantity`: B parameter"""
        return self._b * self._b_units

    @b.setter
    def b(self, value):
        self._b = value.to(self._b_units).magnitude

    _b_units = unit_registry("ppm / (GtC * yr)")
    _b = 1.51 * 10 ** -3

    @property
    def beta(self):
        """:obj:`pint.Quantity`: beta parameter

        This is the fraction of emissions which impact the carbon cycle.
        """
        return self._beta * self._beta_units

    @beta.setter
    def beta(self, value):
        self._beta = value.to(self._beta_units).magnitude

    _beta_units = unit_registry("ppm/GtC")
    _beta = 0.47

    @property
    def sigma(self):
        """:obj:`pint.Quantity`: sigma parameter

        The characteristic response time of the carbon cycle.
        """
        return self._sigma * self._sigma_units

    @sigma.setter
    def sigma(self, value):
        self._sigma = value.to(self._sigma_units).magnitude

    _sigma_units = unit_registry("1/yr")
    _sigma = 2.15 * 10 ** -2

    @property
    def c1(self):
        """:obj:`pint.Quantity`: C1 parameter

        The pre-industrial |CO2| concentration.
        """
        return self._c1 * self._c1_units

    @c1.setter
    def c1(self, value):
        self._c1 = value.to(self._c1_units).magnitude

    _c1_units = unit_registry("ppm")
    _c1 = 290

    @property
    def mu(self):
        """:obj:`pint.Quantity`: mu parameter

        This is like a scaling factor of the radiative forcing due to |CO2| but has
        different units as it is used directly in a temperature response equation rather
        than an energy balance equation.
        """
        return self._mu * self._mu_units

    @mu.setter
    def mu(self, value):
        self._mu = value.to(self._mu_units).magnitude

    # have to initialise like this to avoid ambiguity...
    _mu_tmp = unit_registry.Quantity(8.7 * 10 ** -2, "degC/yr")
    _mu_units = _mu_tmp.units
    _mu = _mu_tmp.magnitude

    @property
    def alpha(self):
        """:obj:`pint.Quantity`: alpha parameter

        The characteristic response time of global-mean temperatures.
        """
        return self._alpha * self._alpha_units

    @alpha.setter
    def alpha(self, value):
        self._alpha = value.to(self._alpha_units).magnitude

    _alpha_units = unit_registry("1/yr")
    _alpha = 1.7 * 10 ** -2

    @property
    def t1(self):
        """:obj:`pint.Quantity`: T1 parameter

        The pre-industrial global-mean temperature.
        """
        # need to better understand
        # pint.errors.OffsetUnitCalculusError: Ambiguous operation with offset unit (degC).
        return unit_registry.Quantity(self._t1, str(self._t1_units))

    @t1.setter
    def t1(self, value):
        self._t1 = value.to(self._t1_units).magnitude

    # have to initialise like this to avoid ambiguity...
    _t1_tmp = unit_registry.Quantity(14.6, "degC")
    _t1_units = _t1_tmp.units
    _t1 = _t1_tmp.magnitude

    @property
    def emissions_idx(self):
        if self._emissions_nan:
            raise ValueError("emissions have not been set yet or contain nan's")

        res = (self._time_current - self._time_start) / self._timestep
        err_msg=(
            "somehow you have reached a point in time which isn't a multiple "
            "of your timeperiod..."
        )
        # assert (res == 0) or (np.abs((res - round(res)) / res) < 10**-5), err_msg
        assert (res == 0) or (-10**-5 < (res - round(res)) < 10**-5)
        assert (
            res >= 0
        ), "somehow you have reached a point in time which is before your starting point..."
        res = round(res)
        try:
            self._emissions[res]
        except IndexError:
            error_msg = (
                "No emissions data available for requested timestep.\n"
                "Requested time: {}\n"
                "Timestep index: {}\n"
                "Length of emissions (remember Python is zero-indexed): {}\n".format(
                    self.time_current, res, len(self.emissions)
                )
            )
            raise OutOfBoundsError(error_msg)

        return res

    def initialise_timeseries(self, driver="emissions"):
        if driver != "emissions":
            raise NotImplementedError("other run modes not implemented yet")

        self.time_current = self.time_start

        initialiser = np.nan * np.zeros_like(self.emissions.magnitude)

        cumulative_emissions_init = deepcopy(initialiser)
        cumulative_emissions_init[0] = 0
        self.cumulative_emissions = unit_registry.Quantity(
            cumulative_emissions_init,
            "GtC"
        )

        concentrations_init = deepcopy(initialiser)
        concentrations_init[0] = 290  # todo: remove hard coding
        self.concentrations = unit_registry.Quantity(
            concentrations_init,
            "ppm"
        )

        temperatures_init = deepcopy(initialiser)
        temperatures_init[0] = 14.6
        self.temperatures = unit_registry.Quantity(
            temperatures_init,
            "degC"
        )

    def run(self, restart=False) -> None:
        """Run the model

        Parameters
        ----------
        restart: bool
            If True, run the model from the first timestep rather than from the value
            of `self.time_current`. This will overwrite any values which have already
            been calculated.
        """
        # super nice that we don't have to write type in docstring when the type is in the function signature
        try:
            self.emissions_idx
        except OutOfBoundsError:
            raise OutOfBoundsError("already run until the end of emissions")

        for _ in range(len(self.emissions)):
            try:
                self.step()
            except OutOfBoundsError:
                break

    def step(self) -> None:
        """Step the model forward to the next point in time"""
        self._step_time()
        self._update_cumulative_emissions()
        self._update_concentrations()
        self._update_temperatures()

    def _step_time(self) -> None:
        self._time_current += self._timestep

    def _update_cumulative_emissions(self) -> None:
        """Update the cumulative emissions to the current timestep"""
        self._check_update_overwrite("_cumulative_emissions")
        self._cumulative_emissions[self.emissions_idx] = (
            self._cumulative_emissions[self.emissions_idx - 1]
            + self._emissions[self.emissions_idx - 1] * self._timestep
        )

    def _update_concentrations(self) -> None:
        """Update the concentrations to the current timestep"""
        self._check_update_overwrite("_concentrations")
        dcdt = (
            self._b * self._cumulative_emissions[self.emissions_idx - 1]
            + self._beta * self._emissions[self.emissions_idx - 1]
            - self._sigma * (self._concentrations[self.emissions_idx - 1] - self._c1)
        )
        self._concentrations[self.emissions_idx] = (
            self._concentrations[self.emissions_idx - 1] + dcdt * self._timestep
        )

    def _update_temperatures(self) -> None:
        """Update the temperatures to the current timestep"""
        self._check_update_overwrite("_temperatures")
        dtdt = self._mu * np.log(
            self._concentrations[self.emissions_idx - 1] / self._c1
        ) - self._alpha * (self._temperatures[self.emissions_idx - 1] - self._t1)
        self._temperatures[self.emissions_idx] = (
            self._temperatures[self.emissions_idx - 1] + dtdt * self._timestep
        )

    def _check_update_overwrite(self, attribute_to_check) -> None:
        """Check if updating the given array will overwrite existing data

        Parameters
        ----------
        attribute_to_check: str
            The attribute of self to check.

        Raises
        ------
        OverwriteError
            If updating the array stored in `attribute_to_check` will overwrite data
            which has already been calculated.
        """
        array_to_check = self.__getattribute__(attribute_to_check)
        if not np.isnan(array_to_check[self.emissions_idx]):
            raise OverwriteError(
                "Stepping {} will overwrite existing data".format(attribute_to_check)
            )
