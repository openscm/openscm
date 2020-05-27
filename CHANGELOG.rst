Changelog
---------

master
******

- (`#190 <https://github.com/openclimatedata/openscm/pull/190>`_) Refactor DICE adapter to make room for PH99 adapter
- (`#187 <https://github.com/openclimatedata/openscm/pull/187>`_) Update timeseries view setting based on `#178 <https://github.com/openclimatedata/openscm/issues/178>`_ and fix bug in time axis overlap checking
- (`#184 <https://github.com/openclimatedata/openscm/pull/184>`_) Fix unit view bug identified in `#177 <https://github.com/openclimatedata/openscm/issues/177>`_
- (`#146 <https://github.com/openclimatedata/openscm/pull/146>`_) Refactor the core interface
- (`#168 <https://github.com/openclimatedata/openscm/pull/168>`_) Fix false positive detection of duplicates when appending timeseries
- (`#166 <https://github.com/openclimatedata/openscm/pull/166>`_) Add usage guidelines
- (`#165 <https://github.com/openclimatedata/openscm/pull/165>`_) Add ``openscm.scenarios`` module with commonly used scenarios
- (`#163 <https://github.com/openclimatedata/openscm/pull/163>`_) Lock ``pyam`` version to pypi versions
- (`#160 <https://github.com/openclimatedata/openscm/pull/160>`_) Update ``setup.py`` so project description is right
- (`#158 <https://github.com/openclimatedata/openscm/pull/158>`_) Add ``ScmDataFrame``, a high-level data and analysis class
- (`#147 <https://github.com/openclimatedata/openscm/pull/147>`_) Remove pyam dependency
- (`#142 <https://github.com/openclimatedata/openscm/pull/142>`_) Add boolean and string parameters
- (`#140 <https://github.com/openclimatedata/openscm/pull/140>`_) Add SARGWP100, AR4GWP100 and AR5GWP100 conversion contexts
- (`#139 <https://github.com/openclimatedata/openscm/pull/139>`_) Add initial definition of parameters
- (`#138 <https://github.com/openclimatedata/openscm/pull/138>`_) Add support for linear point interpolation as well as linear and constant extrapolation
- (`#134 <https://github.com/openclimatedata/openscm/pull/134>`_) Fix type annotations and add a static checker
- (`#133 <https://github.com/openclimatedata/openscm/pull/133>`_) Cleanup and advance timeseries converter
- (`#125 <https://github.com/openclimatedata/openscm/pull/125>`_) Renamed ``timeframes`` to ``timeseries`` and simplified interpolation calculation
- (`#104 <https://github.com/openclimatedata/openscm/pull/104>`_) Define Adapter interface
- (`#92 <https://github.com/openclimatedata/openscm/pull/92>`_) Updated installation to remove notebook dependencies from minimum requirements as discussed in `#90 <https://github.com/openclimatedata/openscm/issues/90>`_
- (`#87 <https://github.com/openclimatedata/openscm/pull/87>`_) Added aggregated read of parameter
- (`#86 <https://github.com/openclimatedata/openscm/pull/86>`_) Made top level of region explicit, rather than allowing access via ``()`` and made requests robust to string inputs
- (`#85 <https://github.com/openclimatedata/openscm/pull/85>`_) Split out submodule for ScmDataFrameBase ``openscm.scmdataframebase`` to avoid circular imports
- (`#83 <https://github.com/openclimatedata/openscm/pull/83>`_) Rename OpenSCMDataFrame to ScmDataFrame
- (`#78 <https://github.com/openclimatedata/openscm/pull/78>`_) Added OpenSCMDataFrame
- (`#44 <https://github.com/openclimatedata/openscm/pull/44>`_) Add timeframes module
- (`#40 <https://github.com/openclimatedata/openscm/pull/40>`_) Add parameter handling in core module
- (`#35 <https://github.com/openclimatedata/openscm/pull/35>`_) Add units module
