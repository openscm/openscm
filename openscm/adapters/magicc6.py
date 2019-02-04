import pymagicc


from ..internal import Adapter
from ..core import Core, ParameterSet


# how to do this intelligently and scalably?
core_climatesensitivity = {"name": "core_climatesensitivity", "unit": "K"}
f2x_co2 = {"name": "core_delq2xco2", "unit": "W / m^2"}
parameters_units = {
    "ecs": core_climatesensitivity,
    "core_climatesensitivity": core_climatesensitivity,
    "core_delq2xco2": f2x_co2,
    "f2xco2": f2x_co2,
}


class MAGICC6(Adapter):
    def __init__(self):
        self._magicc_class = pymagicc.MAGICC6 
        self.magicc = None
        super().__init__()

    def initialize(self, **kwargs) -> None:
        """Initialise the model.

        Parameters
        ----------
        kwargs
            Passed to ``pymagicc.MAGICC6.__init__``
        """
        self.magicc = self._magicc_class(**kwargs)
        self.magicc.__enter__()
        super().initialize()

    def set_drivers(self, parameters: ParameterSet) -> None:
        raise NotImplementedError

    def set_config(self, parameters: ParameterSet) -> None:
        super().set_config(parameters)

        config_dict = {}
        for pname in parameters._root._parameters.keys():
            # need to make this robust to non-scalar views (will
            # also include adding get_array_view, get_boolean_view)
            pview = parameters.get_scalar_view(
                pname,
                parameters._root._name,  # this feels very not robust...
                parameters_units[pname]["unit"]
            )
            config_dict[parameters_units[pname]["name"]] = pview.get()
        
        self.magicc.set_config(**config_dict)

    def run(self) -> Core:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        self.magicc.__exit__()