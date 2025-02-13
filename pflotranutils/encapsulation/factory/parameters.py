__author__ = "Christian Dewey"
__date__ = "Dec 16, 2023"


from dataclasses import dataclass, field
from warnings import warn


from encapsulation.factory.dataProcessing import HDF5Parameters, CrossSectionParameters

class Parameters:
    HDF5Parameters = HDF5Parameters()
    CrossSectionParameters = CrossSectionParameters()

