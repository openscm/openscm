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

    This one box model projects global-mean CO2 concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of CO2 alone.

    Conventions:

    - All fluxes are time period averages and are assumed to be constant throughout
      the time period

    - All state variables are start of time period values


    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    def __init__(self, time_start=0 * unit_registry("s")):
        """Initialise an instance of PH99Model

        Parameters
        ----------
        time_start: :obj:`pint.Quantity`
            Start time of run in seconds since 1970-1-1.

        Side Effects
        ------------
        Sets `self.time_current` to `time_start`
        """
        self.time_start = time_start
        self.time_current = time_start

    _yr = 1 * unit_registry("yr")
    timestep = _yr.to("s")
    """:obj:`pint.Quantity`: Size of timestep in seconds"""
    # TODO: check if this is the right way to describe such a type

    emissions = np.array([np.nan]) * unit_registry("GtC/s")
    """`pint.Quantity` array: Emissions of CO2 in GtC/s"""

    cumulative_emissions = np.array([np.nan]) * unit_registry("GtC")
    """`pint.Quantity` array: Cumulative emissions of CO2 in GtC"""

    concentrations = np.array([np.nan]) * unit_registry("ppm")
    """`pint.Quantity` array: Concentrations of CO2 in ppm"""

    temperatures = unit_registry.Quantity(np.array([np.nan]), "degC")
    """`pint.Quantity` array: Global-mean temperatures in degrees C"""

    b = 1.51 * 10 ** -3 * unit_registry("ppm / (GtC * yr)")
    """:obj:`pint.Quantity`: B parameter in ppm / (GtC yr)"""

    beta = 0.47 * unit_registry("ppm/GtC")
    """:obj:`pint.Quantity`: beta parameter in ppm / GtC

    This is the fraction of emissions which impact the carbon cycle.
    """

    sigma = 2.15 * 10 ** -2 * unit_registry("1/yr")
    """:obj:`pint.Quantity`: sigma parameter in yr^-1

    The characteristic response time of the carbon cycle.
    """
    # TODO: check if you can put latex in docstrings

    c1 = 290 * unit_registry("ppm")
    """:obj:`pint.Quantity`: C1 parameter in ppm

    The pre-industrial CO2 concentration.
    """

    mu = unit_registry.Quantity(8.7 * 10 ** -2, "degC/yr")
    """:obj:`pint.Quantity`: mu parameter in degrees C / yr

    This is like a scaling factor of the radiative forcing due to CO2 but has
    different units as it is used directly in a temperature response equation rather
    than an energy balance equation.
    """

    alpha = 1.7 * 10 ** -2 * unit_registry("1/yr")
    """:obj:`pint.Quantity`: alpha parameter in yr^-1

    The characteristic response time of global-mean temperatures.
    """

    t1 = unit_registry.Quantity(14.6, "degC")
    """:obj:`pint.Quantity`: T1 parameter in degrees C

    The pre-industrial global-mean temperature.
    """

    @property
    def emissions_idx(self):
        if any(np.isnan(self.emissions)):
            raise ValueError("emissions have not been set yet or contain nan's")

        res = (
            ((self.time_current - self.time_start) / self.timestep)
            .to_base_units()
            .magnitude
        )
        np.testing.assert_allclose(
            res, round(res)
        ), "somehow you have reached a point in time which isn't a multiple of your timeperiod..."
        assert (
            res >= 0
        ), "somehow you have reached a point in time which is before your starting point..."
        res = round(res)
        try:
            self.emissions[res]
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
        self.time_current += self.timestep

    def _update_cumulative_emissions(self) -> None:
        """Update the cumulative emissions"""
        self._check_update_overwrite("cumulative_emissions")
        self.cumulative_emissions[self.emissions_idx] = (
            self.cumulative_emissions[self.emissions_idx - 1]
            + self.emissions[self.emissions_idx - 1] * self.timestep
        )

    def _update_concentrations(self) -> None:
        """Update the concentrations"""
        self._check_update_overwrite("concentrations")
        dcdt = (
            self.b * self.cumulative_emissions[self.emissions_idx - 1]
            + self.beta * self.emissions[self.emissions_idx - 1]
            - self.sigma * (self.concentrations[self.emissions_idx - 1] - self.c1)
        )
        self.concentrations[self.emissions_idx] = (
            self.concentrations[self.emissions_idx - 1] + dcdt * self.timestep
        )

    def _update_temperatures(self) -> None:
        """Update the concentrations"""
        self._check_update_overwrite("temperatures")
        dtdt = self.mu * np.log(
            self.concentrations[self.emissions_idx - 1] / self.c1
        ) - self.alpha * (self.temperatures[self.emissions_idx - 1] - self.t1)
        self.temperatures[self.emissions_idx] = (
            self.temperatures[self.emissions_idx - 1] + dtdt * self.timestep
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
