# Expect Exception


Expect Exception allows you to annotate cells in your [Jupyter notebook](https://ipython.org/) that you expect to raise exceptions.  Tracebacks from these exceptions get printed normally, but execution of the notebook does not halt with that cell.  This allows you to run all the cells in the notebook at once, and it makes testing notebooks with exception-raising cells possible.

## Installation

Expect Exception can be installed with pip:
```
pip install git+https://github.com/thedataincubator/expectexception.git@master
```

## Usage

See the [README.ipynb](https://github.com/thedataincubator/expectexception/blob/master/README.ipynb) notebook for usage examples.

## License

Expect Exception is copyright 2016 The Data Incubator, released under the BSD 3-Clause license.  See the LICENSE file for details.  The source is available at https://github.com/thedataincubator/expectexception.
