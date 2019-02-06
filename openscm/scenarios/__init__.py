from os.path import join, dirname, realpath


from ..highlevel import df_append


here = dirname(realpath(__file__))

"""ScmDataFrame: RCP emissions data
"""
rcps = df_append([join(here, "{}_emissions.csv".format(rcp)) for rcp in ["rcp26", "rcp45", "rcp60", "rcp85"]])
