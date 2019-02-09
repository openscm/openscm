import warnings


import numpy as np


from ..internal import Adapter
from ..models import PH99Model
from ..core import Core, ParameterSet
from ..units import unit_registry
from ..errors import NotAnScmParameterError
from ..parameters import _Parameter


class PH99(Adapter):
    """Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999
    This one box model projects global-mean CO2 concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of CO2 alone.
    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    def __init__(self):
        self.model = None
        self.name = "PH99"
        self._ecs = None
        self._hc_per_m2_approx = 30**6 * unit_registry("J / kelvin / m^2")
        super().__init__()

    def initialize(self):
        super().initialize()
        self.model = PH99Model()

    def set_drivers(self, core: Core) -> None:
        self._run_start = core.start_time
        self.model.time_start = core.start_time * unit_registry("s")
        self.model.time_current = core.start_time * unit_registry("s")
        try:
            for cp in core.parameters._root._parameters["Emissions"]._children.values():
                self._set_drivers_from_child_para(core, cp)
        except KeyError:
            raise NotImplementedError("Not ready for non-emissions runs yet")

    def _set_drivers_from_child_para(self, core: Core, parameter: _Parameter):
        if parameter.full_name == ("Emissions", "CO2"):
            pview = core.parameters.get_timeseries_view(
                parameter.full_name,
                ("World",),
                self.model.emissions.units,
                self.model.time_start.to("s").magnitude,
                self.model.timestep.to("s").magnitude
            )
            self.model.emissions = pview.get_series() * self.model.emissions.units
        else:
            warnings.warn("PH99 does not use {}".format(parameter.full_name))

    def set_config(self, parameters: ParameterSet) -> None:
        super().set_config(parameters)
        for key, value in parameters._root._parameters.items():
            self.set_model_parameter(key, value)

    def set_model_parameter(self, para_name, value):
        try:
            modval = value._data * unit_registry(value.info.unit)
            setattr(
                self.model,
                para_name,
                modval.to(getattr(self.model, para_name).units)
            )
        except AttributeError:
            if para_name == "ecs":
                # I'm not sure this is correct, check properly when making proper PR
                self._ecs = modval

                alpha_val = getattr(self.model, "mu") * np.log(2) / modval
                self.model.alpha = alpha_val.to(self.model.alpha.units)
                return

            if para_name == "rf2xco2":
                # I'm not sure this is correct, check properly when making proper PR
                mu_val = modval / self._hc_per_m2_approx
                self.model.mu = mu_val.to(self.model.mu.units)
                if self._ecs is not None:
                    # reset alpha too as it depends on mu
                    alpha_val = getattr(self.model, "mu") * np.log(2) / self._ecs
                    self.model.alpha = alpha_val.to(self.model.alpha.units)
                return

            raise NotAnScmParameterError(
                "{} is not a PH99 parameter".format(para_name)
            )

    def run(self) -> None:
        self.model.initialise_timeseries()
        self.model.run()

        st = self._run_start
        et_raw = (
            self.model.time_current
            - self.model.time_start
        ).to("s").magnitude
        et = int(st + et_raw)

        results = Core("PH99", st, et)
        results.period_length = self.model.timestep.to("s").magnitude

        for att in dir(self.model):
            # all time parameters captured in parameterset output
            if not att.startswith(("_", "time", "emissions_idx")):
                value = getattr(self.model, att)
                if callable(value):
                    continue

                name = self._get_openscm_name(att)
                magnitude = value.magnitude

                if isinstance(magnitude, np.ndarray):
                    if name == ("Surface Temperature"):
                        # this is where reference period etc. is important
                        value = value - self.model.t1
                        magnitude = value.magnitude

                    results.parameters.get_writable_timeseries_view(
                        name,
                        ("World",),
                        str(value.units),
                        self.model.time_start.magnitude,
                        self.model.timestep.magnitude,
                    ).set_series(magnitude)
                else:
                    results.parameters.get_writable_scalar_view(
                        name,
                        ("World"),
                        str(value.units)
                    ).set(magnitude)

        ecs = (self.model.mu * np.log(2) / self.model.alpha).to("K")
        results.parameters.get_writable_scalar_view(
            ("ecs",),
            ("World"),
            str(ecs.units)
        ).set(ecs.magnitude)


        rf2xco2 = self.model.mu * self._hc_per_m2_approx
        results.parameters.get_writable_scalar_view(
            ("rf2xco2",),
            ("World"),
            str(rf2xco2.units)
        ).set(rf2xco2.magnitude)

        return results

    def _get_openscm_name(self, name):
        mappings = {
            "concentrations": ("Atmospheric Concentrations", "CO2"),
            "cumulative_emissions": ("Cumulative Emissions", "CO2"),
            "emissions": ("Emissions", "CO2"),
            "temperatures": ("Surface Temperature"),
        }
        try:
            return mappings[name]
        except KeyError:
            return (name,)

    def step(self) -> None:
        self.model.step()

    def shutdown(self) -> None:
        super().shutdown()
