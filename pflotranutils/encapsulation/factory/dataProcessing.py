__author__ = "Christian Dewey"
__date__ = "Dec 16, 2023"

import os 
import sys
import h5py 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


from dataclasses import dataclass, field
from warnings import warn

from matplotlib import axes

import sys
from pathlib import Path

import pandas as pd

class HDF5Parameters:

	def __init__(self):

		self.file_location = None


class CrossSectionParameters:

	def __init__(self):

		self.time_unit = None
		self.perpendicular_axis = 'x'