"""
Models maintained within the OpenSCM project.
 These are generally small, 'toy' models which do not require a separate package
to implement (i.e. they have few or no dependencies). They follow the following
general guidelines:
 - a model should take in a time start and have a timeperiod attribute. This avoids:
     - requiring models to interpolate internally, that should be somewhere else in
      pre-processing.
     - having to worry about irregular timesteps
         - with Pint, a month is just 1/12 of a year so that is a regular
          timestep
         - if people want to convert back to human calendars later, they can do so but
          that should be a pre/post-processing step
"""
from .ph99 import PH99Model  # noqa: F401
