from . import Adapter

YEAR = 365 * 24 * 60 * 60  # example time step length as used below

# Zeb's scribbles as he does this (may be helpful for later):
#   - MAGICC doesn't need to have a constant timestep, hence can't directly copy the
#     examples of DICE or PH99
#   -

class MAGICC7(Adapter):
    # _initialize_model needs to do initalize (create copy of MAGICC)
    # _shutdown needs to do the cleanup (delete copy of MAGICC)
    # _set_model_from_parameters needs to re-write the input files
    # reset needs to delete everything in `out` and cleanup the output parameters
    # _run just calls the .run method
    # _step is not implemented
    # _get_time_points can just use the openscm parameters (only need to pass back
    #   down at set_model_from_parameters calls)
    # _update_model just calls MAGICC's update_config method
    # name is MAGICC7 (as it's what is used in OpenSCM parameters)
    pass
