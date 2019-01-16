import sys
from IPython.core.magic import Magics, magics_class, cell_magic


class ExceptionExpected(Exception):
    pass


@magics_class
class ExceptionMagics(Magics):

    # Note: This gets made into an instance method on the IPython shell
    @staticmethod
    def custom_handler(self, etype, value, tb, tb_offset=None):
        # Save these so that we can use the debugger.
        sys.last_type = etype
        sys.last_value = value
        sys.last_traceback = tb

        stb = self.InteractiveTB.structured_traceback(etype, value, tb, tb_offset)
        for line in stb:
            print(line)
        return stb

    @cell_magic
    def expect_exception(self, line, cell):
        self.run_cell(line, cell, True)

    @cell_magic
    def ignore_exception(self, line, cell):
        self.run_cell(line, cell, False)

    def run_cell(self, line, cell, exception_required):
        if not line:
            line = "Exception"
        try:
            exception = self.shell.ev(line)
        except (SyntaxError, NameError):
            raise NameError("Could not evaluate name of expected exception")
        if not (isinstance(exception, type) and issubclass(exception, Exception)):
            raise TypeError("Argument to expect_exception must be an Exception")

        old_CustomTB = self.shell.CustomTB
        old_custom_exceptions = self.shell.custom_exceptions
        try:
            self.shell.set_custom_exc((exception,), self.custom_handler)
            result = self.shell.run_cell(cell)
            if exception_required and not result.error_in_exec:
                raise ExceptionExpected("This cell did not raise the expected %s." % line)
        finally:
            self.shell.CustomTB = old_CustomTB
            self.shell.custom_exceptions = old_custom_exceptions
