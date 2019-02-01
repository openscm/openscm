import functools


def str_to_list(base_func):
    """Make a function which expects list input able to handle str input too
    """
    # a good idea according to
    # https://realpython.com/primer-on-python-decorators/#simple-decorators...
    @functools.wraps(base_func)
    def flexible_to_str_input(*args):
        new_args = tuple([
            [a] if isinstance(a, str) else a
            for a in args
        ])

        return base_func(*new_args)

    return flexible_to_str_input
