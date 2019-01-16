from .excpectexceptionmagic import ExceptionMagics, ExceptionExpected

get_ipython().register_magics(ExceptionMagics)
