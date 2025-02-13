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

from pflotranutils.encapsulation.factory.parameters import Parameters

class HDF5Output:
	def __init__(self, file_location):

		self._init_settings()
		
		if isinstance(file_location, str):
			file_path = Path(file_location)

		else:
			file_path = file_location
		
		self.h5_data = h5py.File((str(file_path)),'r')

		self.data_groups = list(self.h5_data.keys())

		self.get_times()
		self.get_components()

	def _init_settings(self):

		self.h5_parameters = Parameters.HDF5Parameters

	def get_components(self):

		h5_data = self.h5_data
		self.component_list = list(h5_data[self._time_0_group].keys())
	
	def print_components(self,include=None, exclude=None):
		if include != None:
			printlist = [i for i in self.component_list if include in i]
		else:
			printlist = self.component_list

		if exclude != None:
			printlist = [i for i in printlist if exclude not in i]

		for c in printlist:
			print(c)

	def get_times(self):

		groups = self.data_groups

		self.times = np.sort(np.array([[float(gg[7:18]) for gg in groups if 'Time' in gg]]))
		self.times = np.reshape(self.times,(-1,))
		self._get_time_0_group()

	def _get_time_0_group(self):

		time_0 = self.times[0]
		time_str = f'Time:  {time_0:.5E}'
		groups = self.data_groups
		gg_i = [gi for gi, group in zip(range(len(groups)), groups) if time_str in group]
		gg = groups[gg_i[0]]

		self._time_0_group = gg

	def get_time_t_group(self,time_t):

		
		time_str = f'Time:  {time_t:.5E}'
		groups = self.data_groups
		gg_t = [gi for gi, group in zip(range(len(groups)), groups) if time_str in group]
		if len(gg_t) == 0:
			print('No output for that time. Only output for:')
			for t in self.times[0]:
				print(t)
			return None
		else:
			return groups[gg_t[0]]

		