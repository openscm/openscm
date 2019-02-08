from os.path import join


import pymagicc


from ..internal import Adapter
from ..core import Core, ParameterSet
from ..parameters import ParameterType
from ..errors import NotAnScmParameterError
from ..highlevel import convert_core_to_scmdataframe
from ..utils import round_to_nearest_year


# how to do this intelligently and scalably?
core_climatesensitivity = {
    "name": "core_climatesensitivity",
    "type": ParameterType.SCALAR,
    "unit": "K",
}
f2x_co2 = {"name": "core_delq2xco2", "type": ParameterType.SCALAR, "unit": "W / m^2"}
parameters_magicc = {
    "ecs": core_climatesensitivity,
    "core_climatesensitivity": core_climatesensitivity,
    "core_delq2xco2": f2x_co2,
    "f2xco2": f2x_co2,
    "co2_tempfeedback_switch": {
        "name": "co2_tempfeedback_switch",
        "type": ParameterType.BOOLEAN,
    },
    "gen_sresregions2nh": {
        "name": "gen_sresregions2nh",
        "type": ParameterType.ARRAY,
        "unit": "dimensionless",
    },
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

    def set_drivers(self, core: Core) -> None:
        # fix once Jared has Pymagicc working with ScmDataFrame
        scen = pymagicc.io.MAGICCData(convert_core_to_scmdataframe(core).timeseries())
        scen["time"] = scen["time"].apply(round_to_nearest_year)
        scen.write(
            join(self.magicc.run_dir, self.magicc._scen_file_name),
            self.magicc.version
        )

    def set_config(self, parameters: ParameterSet) -> None:
        super().set_config(parameters)

        config_dict = {}
        for pname, pval in parameters._root._parameters.items():
            try:
                n, v = self._get_config_dict_name_value(parameters, pname, pval)
                config_dict[n] = v
            except KeyError:
                msg = "{} is not a MAGICC6 parameter".format(pname)
                raise NotAnScmParameterError(msg)

        self.magicc.set_config(**config_dict)

    def _get_config_dict_name_value(self, parameters, pname, pval):
        # TODO: add better region handling for parameters
        # In MAGICC they're all World so doesn't matter yet (arrays are the
        # regional parameters kind of...)
        if pval.info._type == ParameterType.SCALAR:
            pview = parameters.get_scalar_view(
                pname, (pval.info.region), parameters_magicc[pname]["unit"]
            )
            return parameters_magicc[pname]["name"], pview.get()
        elif pval.info._type == ParameterType.ARRAY:
            pview = parameters.get_array_view(
                pname, (pval.info.region), parameters_magicc[pname]["unit"]
            )
            return parameters_magicc[pname]["name"], list(pview.get())
        elif pval.info._type == ParameterType.BOOLEAN:
            pview = parameters.get_boolean_view(pname, (pval.info.region))
            return parameters_magicc[pname]["name"], pview.get()
        else:
            raise NotImplementedError

    def run(self) -> Core:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        self.magicc.__exit__()
