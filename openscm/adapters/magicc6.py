from os.path import join


import numpy as np
import pymagicc


from ..internal import Adapter
from ..core import Core, ParameterSet
from ..parameters import ParameterType
from ..errors import NotAnScmParameterError
from ..highlevel import convert_core_to_scmdataframe, convert_scmdataframe_to_core
from ..utils import round_to_nearest_year


# how to do this intelligently and scalably?
_mapping = {"ecs": "core_climatesensitivity", "rf2xco2": "core_delq2xco2"}
openscm_para_magicc_mapping = {}
for k, v in _mapping.items():
    openscm_para_magicc_mapping[k] = v
    openscm_para_magicc_mapping[v] = k

parameters_magicc = {
    "core_climatesensitivity": {"type": ParameterType.SCALAR, "unit": "kelvin"},
    "core_delq2xco2": {"type": ParameterType.SCALAR, "unit": "W / m^2"},
    "co2_tempfeedback_switch": {"type": ParameterType.BOOLEAN},
    "gen_sresregions2nh": {"type": ParameterType.ARRAY, "unit": "dimensionless"},
}


class MAGICC6(Adapter):
    def __init__(self):
        self._magicc_class = pymagicc.MAGICC6
        self.magicc = None
        self.name = "MAGICC{}".format(self._magicc_class.version)
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
        scen.set_meta("SET", name="todo")
        scen.write(
            join(self.magicc.run_dir, self.magicc._scen_file_name), self.magicc.version
        )

    def set_config(self, parameters: ParameterSet) -> None:
        super().set_config(parameters)

        config_dict = {"file_emisscen": self.magicc._scen_file_name}
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
        if pname in parameters_magicc:
            magicc_name = pname
        else:
            magicc_name = openscm_para_magicc_mapping[pname]

        if pval.info._type == ParameterType.SCALAR:
            pview = parameters.get_scalar_view(
                pname, (pval.info.region), parameters_magicc[magicc_name]["unit"]
            )
            return magicc_name, pview.get()
        elif pval.info._type == ParameterType.ARRAY:
            pview = parameters.get_array_view(
                pname, (pval.info.region), parameters_magicc[magicc_name]["unit"]
            )
            return magicc_name, list(pview.get())
        elif pval.info._type == ParameterType.BOOLEAN:
            pview = parameters.get_boolean_view(pname, (pval.info.region))
            return magicc_name, pview.get()
        else:
            raise NotImplementedError

    def run(self) -> Core:
        res = self.magicc.run(
            startyear=1765,
            endyear=2500,
            only=["Emissions|CO2|MAGICC Fossil and Industrial", "Surface Temperature"],
        )  # hard code for now
        results = convert_scmdataframe_to_core(res, climate_model=self.name)
        for k, v in res.metadata["parameters"]["allcfgs"].items():
            try:
                if parameters_magicc[k]["type"] == ParameterType.SCALAR:
                    set_val = v
                    pview = results.parameters.get_writable_scalar_view(
                        k, ("World",), parameters_magicc[k]["unit"]
                    )
                    pview.set(set_val)
                    if k in openscm_para_magicc_mapping:
                        other_view = results.parameters.get_writable_scalar_view(
                            openscm_para_magicc_mapping[k],
                            ("World",),
                            parameters_magicc[k]["unit"],
                        )
                        other_view.set(set_val)
                elif parameters_magicc[k]["type"] == ParameterType.ARRAY:
                    set_val = np.array(v)
                    pview = results.parameters.get_writable_array_view(
                        k, ("World",), parameters_magicc[k]["unit"]
                    )
                    pview.set(set_val)
                    if k in openscm_para_magicc_mapping:
                        other_view = results.parameters.get_writable_array_view(
                            openscm_para_magicc_mapping[k],
                            ("World",),
                            parameters_magicc[k]["unit"],
                        )
                        other_view.set(set_val)
                elif parameters_magicc[k]["type"] == ParameterType.BOOLEAN:
                    set_val = bool(v)
                    pview = results.parameters.get_writable_boolean_view(k, ("World",))
                    pview.set(set_val)
                    if k in openscm_para_magicc_mapping:
                        other_view = results.parameters.get_writable_boolean_view(
                            openscm_para_magicc_mapping[k], ("World",)
                        )
                        other_view.set(set_val)
                else:
                    raise NotImplementedError
            except KeyError:
                continue
        return results

    def step(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        self.magicc.__exit__()
