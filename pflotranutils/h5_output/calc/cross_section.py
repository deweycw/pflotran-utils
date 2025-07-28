__author__ = "Christian Dewey"
__date__ = "Dec 16, 2023"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoLocator)
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

from dataclasses import dataclass, field
from warnings import warn
import warnings
warnings.filterwarnings('ignore')

import sys

import seaborn as sns

from pflotranutils.encapsulation.factory.parameters import Parameters
from pflotranutils.h5_output.factory.h5_output_class import HDF5Output

class CrossSection(HDF5Output):
	
	def __init__(self, file_location,perpendicular_axis):
		super().__init__(file_location)


		self.perpendicular_axis = perpendicular_axis

		self._init_settings()


	def _init_settings(self):

		self.xsection_parameters = Parameters.CrossSectionParameters


	def get_cells(self, perp_loc = None, print_ds = False):
		# perp loc in m 
		h5_data = self.h5_data

		self.xs = np.array(h5_data['Coordinates']['X [m]'])
		self.ys = np.array(h5_data['Coordinates']['Y [m]'])
		self.zs = np.array(h5_data['Coordinates']['Z [m]'])

		dx = self.xs[1] - self.xs[0]
		dy = self.ys[1] - self.ys[0]
		dz = self.zs[1] - self.zs[0]
		self.ds= (dx,dy,dz)

		self.domain_dimension = (len(self.xs)-1, len(self.ys)-1, len(self.zs)-1)
		self.cross_section_dimension = [i for i in self.domain_dimension]
		start_indices = [0, 0, 0]
		end_indices = [i for i in self.domain_dimension ]

		delta = u'Δ'
		dim = 'n cells'
		
		dims = self.domain_dimension
		if print_ds:
			print(f'{delta:7}  {self.ds[0]:4} {self.ds[1]:4} {self.ds[2]:4}')
			print(f'{dim:6}  {dims[0]:4} {dims[1]:4} {dims[2]:4}')
		if self.perpendicular_axis == 'x':
			if self.domain_dimension[0] > 1:
				start = int(perp_loc / dx)
			else:
				start = 0
			start_indices[0] = start
			end_indices[0] = start+1
			del self.cross_section_dimension[0]
			
		elif self.perpendicular_axis == 'y':
			if self.domain_dimension[1] > 1:
				start = int(perp_loc / dy)
			else:
				start = 0
			start_indices[1] = start
			end_indices[1] = start+1
			del self.cross_section_dimension[1]

		elif self.perpendicular_axis == 'z':
			if self.domain_dimension[2] > 1:
				start = int(perp_loc / dz)
			else:
				start = 0
			start_indices[2] = start
			end_indices[2] = start+1
			del self.cross_section_dimension[2]
		else:
			print('axis not recognized')
		

		self.cross_section_cells = [(i,j) for i, j in zip(start_indices,end_indices)]
		if print_ds:
			print(f'{self.perpendicular_axis}-axis is perpendicular\ncross-section cells: x-{self.cross_section_cells[0]}, y-{self.cross_section_cells[1]}, z-{self.cross_section_cells[2]}')

	def get_material_ids(self,show_inactive=False, locs=None):

		self.plot_at_time(0,'Material_ID',show_inactive,locs=locs)

	def plot_velocity_at_time(self,time_t,perpendicular_axis, ax, show_unsat):   #,show_inactive=False,show_unsat=True

		def get_oriented_vel_mats(inds, perpendicular_axis, show_unsat):

			if perpendicular_axis == 'x':
				component_in_x = "Liquid Y-Velocity [m_per_hr]"
				component_in_y = "Liquid Z-Velocity [m_per_hr]"
			elif perpendicular_axis == 'y':
				component_in_x = "Liquid X-Velocity [m_per_hr]"
				component_in_y = "Liquid Z-Velocity [m_per_hr]"		
			elif perpendicular_axis == 'z':
				component_in_x = "Liquid X-Velocity [m_per_hr]"
				component_in_y = "Liquid Y-Velocity [m_per_hr]"

			full_set_x = np.array(h5_data[t_group][component_in_x])
			full_set_y = np.array(h5_data[t_group][component_in_y])

			cross_set_x = full_set_x[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
			cross_set_y = full_set_y[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]

			flat_shape_x = [i for i in np.shape(cross_set_x) if i != 1]	 
			flat_shape_y = [i for i in np.shape(cross_set_y) if i != 1]	 

			flattened_cross_set_x = np.reshape(cross_set_x,flat_shape_x)
			flattened_cross_set_y = np.reshape(cross_set_y,flat_shape_y)

			def get_oriented_set(flattened_cross_set):
				if len(np.shape(flattened_cross_set)) == 2:
					oriented_set = np.fliplr(flattened_cross_set).T
				else:
					oriented_set = np.flip(np.reshape(flattened_cross_set,(len(flattened_cross_set),1)))
				return oriented_set
			
			oriented_vel_set_x = get_oriented_set(flattened_cross_set_x)
			oriented_vel_set_y = get_oriented_set(flattened_cross_set_y)


			def get_only_saturated(flat_shape, oriented_set, show_unsat):

				matid_set = np.array(h5_data[t_group]['Material_ID'])
				matid_cross_set = matid_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
				matid_flattened_cross_set = np.reshape(matid_cross_set,flat_shape)
				matid_oriented_set = np.fliplr(matid_flattened_cross_set).T

				sat_full_set = np.array(h5_data[t_group]['Liquid_Saturation'])
				sat_cross_set = sat_full_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
				sat_flattened_cross_set = np.reshape(sat_cross_set,flat_shape)
				sat_oriented_set = np.fliplr(sat_flattened_cross_set).T
				sat_oriented_set[np.where(matid_oriented_set == 0)]=1
				sat_oriented_set[np.where(sat_oriented_set != 1)] = 0

				masked_set = np.ma.masked_where((sat_oriented_set == 0) | (matid_oriented_set == 0), oriented_set)

				return masked_set
			

			if show_unsat == False:
				
				x_vel_set = get_only_saturated(flat_shape_x, oriented_vel_set_x, show_unsat)
				y_vel_set = get_only_saturated(flat_shape_y, oriented_vel_set_y, show_unsat)

				return x_vel_set, y_vel_set
			
			else:
	
				return oriented_vel_set_x, oriented_vel_set_y

			
		h5_data = self.h5_data
		t_group = self.get_time_t_group(time_t)
		
		if t_group is not None:

			inds = self.cross_section_cells
	
			oriented_vel_set_x, oriented_vel_set_y = get_oriented_vel_mats(inds,perpendicular_axis, show_unsat)

			
			nx = len(oriented_vel_set_x[0,:])
			ny = len(oriented_vel_set_x[:,0])

			m = np.meshgrid(range(nx), range(ny))

			xg = m[0][0]
			yg = m[1][:,0]

			mag_mat = np.sqrt(oriented_vel_set_x**2 + oriented_vel_set_y**2)

			max_mag = np.max(mag_mat)
			lwd = mag_mat / max_mag *0.8
			ax.streamplot(xg, yg, oriented_vel_set_x, -1* oriented_vel_set_y, color= 'white', #mag_mat, cmap = 'cividis',
				 arrowstyle= '->',linewidth = lwd,arrowsize = 0.8,  density=(0.15,0.45), broken_streamlines=True)
			
			'''if self.perpendicular_axis == 'x':
				di = self.ds[1]
				dj = self.ds[2]
			elif self.perpendicular_axis == 'y':
				di = self.ds[0]
				dj = self.ds[2]	
			else:
				di = self.ds[0]
				dj = self.ds[1]

			ax.xaxis.set_major_locator(AutoLocator())
			x_majortick_locs = ax.get_xticks()
			x_majortick_lbls = [f'{i*di:.0f}' for i in x_majortick_locs]
			ax.set_xticks(x_majortick_locs)
			ax.set_xticklabels(x_majortick_lbls)

			ax.yaxis.set_major_locator(AutoLocator())
			y_majortick_locs = ax.get_yticks()
			y_majortick_lbls = [f'{j*dj:.1f}' for j in y_majortick_locs]
			ax.set_yticks(y_majortick_locs)
			ax.set_yticklabels(y_majortick_lbls)

			ax.set_xlim(-1,np.shape(oriented_vel_set_y)[1]+1)
			ax.set_ylim(np.shape(oriented_vel_set_y)[0]+1,-1)
			ax.set_xlabel('Distance (m)')
			ax.set_ylabel('Depth below datum (m)')'''

			return ax

	def get_min_max(self,df,component):
		if 'pH' in component:
			mindf = np.amin(df[np.isfinite(df)])
		elif 'dG-rxn' in component:
			mindf = np.amin(df[np.isfinite(df)])
		elif 'Rate' in component:
			#df = np.log10(-1 * df)
			mindf = np.amin(df[np.isfinite(df)])
		else:
			mindf = 0
		maxdf = np.amax(df[np.isfinite(df)])
		return mindf,maxdf
	
	def plot_at_time(self,  
					time_t,
					component,
					show_inactive=False,
					show_unsat=True,
					locs=None,
					ax=None,
					min=None,
					max=None,
					unit=None,
					exp_in_unit_label=True,
					plot_vel=False):

		h5_data = self.h5_data
		t_group = self.get_time_t_group(time_t)
			
		if t_group is not None:
			full_set = np.array(h5_data[t_group][component])

			inds = self.cross_section_cells
			
			cross_set = full_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]

			flat_shape = [i for i in np.shape(cross_set) if i != 1]	 

			if np.shape(cross_set).count(1) > 1: # if a 1D model domain

				flattened_cross_set = np.reshape(cross_set,(1,flat_shape[0]))
				if len(np.shape(flattened_cross_set)) == 2:
					oriented_set = np.fliplr(flattened_cross_set).T
				else:
					oriented_set = np.flip(np.reshape(flattened_cross_set,(len(flattened_cross_set),1)))

				if show_inactive:
					plot_set = oriented_set
				else:
					matid_set = np.array(h5_data[t_group]['Material_ID'])
					matid_cross_set = matid_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
					matid_flattened_cross_set = np.reshape(matid_cross_set,(1,flat_shape[0]))
					matid_oriented_set = np.fliplr(matid_flattened_cross_set).T
					plot_set = np.ma.masked_where(matid_oriented_set == 0, oriented_set)
					
				if show_unsat is False:
					sat_full_set = np.array(h5_data[t_group]['Liquid_Saturation'])
					sat_cross_set = sat_full_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
					sat_flattened_cross_set = np.reshape(sat_cross_set,(1, flat_shape[0]))
					sat_oriented_set = np.fliplr(sat_flattened_cross_set).T 
					sat_oriented_set[np.where(matid_oriented_set == 0)]=1
					sat_oriented_set[np.where(sat_oriented_set != 1)] = 0
					unsat_oriented_set = np.ma.masked_where(sat_oriented_set == 1, sat_oriented_set)
					plot_set = np.ma.masked_where((sat_oriented_set == 0) | (matid_oriented_set == 0), oriented_set)

				else:
					unsat_oriented_set = None
					sat_oriented_set = None
		
				self._plot_1d_domain(flattened_cross_set = flattened_cross_set,
														component=component,
														time_t=time_t,
														show_unsat=show_unsat,
														plot_set = plot_set,
														unsat_oriented_set = unsat_oriented_set,
														sat_oriented_set= sat_oriented_set,
														locs = locs,
														ax = ax,
														min = min,
														max = max,
														unit = unit,
														exp_in_unit_label = exp_in_unit_label,
														plot_vel = plot_vel)
				
			else:

				flattened_cross_set = np.reshape(cross_set,flat_shape)
				if len(np.shape(flattened_cross_set)) == 2:
					oriented_set = np.fliplr(flattened_cross_set).T
				else:
					oriented_set = np.flip(np.reshape(flattened_cross_set,(len(flattened_cross_set),1)))
				
				if show_inactive:
					plot_set = oriented_set
				else:
					matid_set = np.array(h5_data[t_group]['Material_ID'])
					matid_cross_set = matid_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
					matid_flattened_cross_set = np.reshape(matid_cross_set,flat_shape)
					matid_oriented_set = np.fliplr(matid_flattened_cross_set).T
					plot_set = np.ma.masked_where(matid_oriented_set == 0, oriented_set)
					
				if self.perpendicular_axis == 'x':
					di = self.ds[1]
					dj = self.ds[2]
				elif self.perpendicular_axis == 'y':
					di = self.ds[0]
					dj = self.ds[2]	
				else:
					di = self.ds[0]
					dj = self.ds[1]
				
				showax = False
				if ax == None:
					showax = True
					fig, ax = plt.subplots()	
				
				if unit == None:
					unit_factor = 1
				elif unit == 'uM':
					unit_factor = 1e6
				elif unit == 'mM':
					unit_factor = 1e3

				plot_set = plot_set * unit_factor	
				
				mindf,maxdf = self.get_min_max(plot_set,component)
				if min != None:
					mindf = min
				if max != None: 
					maxdf = max
				print('min value for %s: %.3E' %(component, mindf))
				print('max value for %s: %.3E' %(component, maxdf))

				if show_unsat is False:
					unsat_cmap = ListedColormap(["black", "white"])
					sat_full_set = np.array(h5_data[t_group]['Liquid_Saturation'])
					sat_cross_set = sat_full_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
					sat_flattened_cross_set = np.reshape(sat_cross_set,flat_shape)
					sat_oriented_set = np.fliplr(sat_flattened_cross_set).T
					sat_oriented_set[np.where(matid_oriented_set == 0)]=1
					sat_oriented_set[np.where(sat_oriented_set != 1)] = 0
					unsat_oriented_set = np.ma.masked_where(sat_oriented_set == 1, sat_oriented_set)
					ax.imshow(unsat_oriented_set,cmap=unsat_cmap,alpha=0.5)
					plot_set = np.ma.masked_where((sat_oriented_set == 0) | (matid_oriented_set == 0), oriented_set)
					norm = mcolors.Normalize(vmin=mindf, vmax=maxdf) 
					ax.imshow(plot_set,cmap='viridis',norm=norm)
					if plot_vel:
						self.plot_velocity_at_time(time_t,'x',ax,show_unsat)
				
				else:
					norm = mcolors.Normalize(vmin=mindf, vmax=maxdf) 
					ax.imshow(plot_set,cmap='viridis',norm=norm)
					if plot_vel:
						self.plot_velocity_at_time(time_t,'x',ax,show_unsat)

				if locs != None:
					xobs = [i[0]/0.5 for i in locs]
					yobs = [i[1]/0.1 for i in locs]
					ax.scatter(xobs, yobs,marker='o',color='r' )


				ax.xaxis.set_major_locator(AutoLocator())
				x_majortick_locs = ax.get_xticks()
				x_majortick_lbls = [f'{i*di:.0f}' for i in x_majortick_locs]
				ax.set_xticks(x_majortick_locs)
				ax.set_xticklabels(x_majortick_lbls)

				
				ax.yaxis.set_major_locator(AutoLocator())
				y_majortick_locs = ax.get_yticks()
				y_majortick_lbls = [f'{j*dj:.1f}' for j in y_majortick_locs]
				ax.set_yticks(y_majortick_locs)
				ax.set_yticklabels(y_majortick_lbls)

				ax.set_xlim(-1,np.shape(plot_set)[1]+1)
				ax.set_ylim(np.shape(plot_set)[0]+1,-1)
				ax.set_xlabel('Distance (m)')
				ax.set_ylabel('Depth below datum (m)')
				ax.set_title(f'{component} at {time_t:.1f} h', size = 10)

				sns.despine()
				
				if showax:
					self._makeColorBars(fig,mindf,maxdf,exp_in_unit_label)

					fig.tight_layout()

					plt.show()
				else:
					return (ax, mindf, maxdf)

	def _plot_1d_domain(self, 
						flattened_cross_set, 
						component, 
						time_t, 
						show_unsat=True, 
						unsat_oriented_set=None,
						sat_oriented_set = None,
						plot_set=None,
						locs=None,
						ax=None, 
						min=None, 
						max=None, 
						unit=None, 
						exp_in_unit_label = True,
						plot_vel=False):

		showax = False
		if ax == None:
			showax = True
			fig, ax = plt.subplots()	

		if self.perpendicular_axis == 'x':
			di = self.ds[1]
			dj = self.ds[2]
		elif self.perpendicular_axis == 'y':
			di = self.ds[0]
			dj = self.ds[2]	
		else:
			di = self.ds[0]
			dj = self.ds[1]
		

	
		if component == 'pH':
			unit = None
		
		if unit == None:
			unit_factor = 1
		elif unit == 'uM':
			unit_factor = 1e6
		elif unit == 'mM':
			unit_factor = 1e3

		if len(np.shape(flattened_cross_set)) == 2:
			oriented_set = np.fliplr(flattened_cross_set).T
		else:
			oriented_set = np.flip(np.reshape(flattened_cross_set,(len(flattened_cross_set),1)))
		

		plot_set = oriented_set * unit_factor	
		
		mindf,maxdf = self.get_min_max(plot_set,component)

		if min != None:
			mindf = min
		if max != None: 
			maxdf = max
		
		print('min value for %s: %.3E' %(component, mindf))
		print('max value for %s: %.3E' %(component, maxdf))

		if show_unsat is False:
			unsat_cmap = ListedColormap(["black", "white"])
			norm = mcolors.Normalize(vmin=mindf, vmax=maxdf) 
			ax.imshow(unsat_oriented_set,cmap=unsat_cmap,alpha=0.5,aspect=0.2)
			plot_set = np.ma.masked_where(sat_oriented_set == 0 , plot_set)
			ax.imshow(plot_set,cmap='viridis',norm=norm,aspect=0.2)

		else:
			norm = mcolors.Normalize(vmin=mindf, vmax=maxdf) 
			ax.imshow(plot_set,cmap='viridis',norm=norm,aspect=0.2)

		if plot_vel:
			self.plot_velocity_at_time(time_t,'x',ax,show_unsat)

		if locs != None:
			xobs = [i[0]/0.5 for i in locs]
			yobs = [i[1]/0.1 for i in locs]
			ax.scatter(xobs, yobs,marker='o',color='r' )


		x_majortick_locs = ax.get_xticks()
		x_majortick_lbls = [' ' for i in x_majortick_locs]
		#ax.set_xticks(x_majortick_locs)
		ax.set_xticklabels(x_majortick_lbls)
		
		ax.yaxis.set_major_locator(AutoLocator())
		y_majortick_locs = ax.get_yticks()
		y_majortick_lbls = [f'{j*dj:.1f}' for j in y_majortick_locs]
		ax.set_yticks(y_majortick_locs)
		ax.set_yticklabels(y_majortick_lbls)
		print(np.shape(plot_set)[1]+1)
		ax.set_xlim(-1,np.shape(plot_set)[1])
		ax.set_ylim(np.shape(plot_set)[0]+1,-1)
		#ax.set_xlabel('Distance (m)')
		ax.set_ylabel('Depth below surface (m)')
		ax.set_title(f'{component} at {time_t:.1f} h', size = 10)

		sns.despine()
		
		if showax:
			self._makeColorBars(fig,mindf,maxdf,exp_in_unit_label)

			fig.tight_layout()

			plt.show()
		else:
			return (ax, mindf, maxdf)
		

	def _makeColorBars(self, fig, mindf, maxdf, exp_in_unit_label):
		cbar_ax = fig.add_axes([1.01, 0.32, 0.02, 0.3])
		nValues = np.arange(mindf,maxdf)
		norm = mcolors.Normalize(vmin=mindf, vmax=maxdf) 
		scalarmappaple = cm.ScalarMappable(norm = norm, cmap='viridis')
		scalarmappaple.set_array(nValues)
		
		cb = fig.colorbar(scalarmappaple, orientation='vertical',cax=cbar_ax,shrink=0.5)


		if exp_in_unit_label:
			cb_labels = [f'{i:.1E}' for i in cb.ax.get_yticks()]
		else:
			cb_labels = [f'{i:.1f}' for i in cb.ax.get_yticks()] 

		cb.ax.set_yticklabels( cb_labels,size = 8 )
		return cb


	def get_component_at_cell(self,component,cell_loc,time_t):

		h5_data = self.h5_data
		t_group = self.get_time_t_group(time_t)
			
		if t_group is not None:
			full_set = np.array(h5_data[t_group][component])

			inds = self.cross_section_cells
			
			cross_set = full_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]

			flat_shape = [i for i in np.shape(cross_set) if i != 1]	 
			flattened_cross_set = np.reshape(cross_set,flat_shape)

			if len(flat_shape) > 1:
				
				oriented_set = np.fliplr(flattened_cross_set).T
			
				return oriented_set[cell_loc[1],cell_loc[0]]
			else:
				return flattened_cross_set[cell_loc[0]]
		
	def get_history_at_m_coords(self,component,meter_coords: tuple):
		
		if self.perpendicular_axis == 'x':
			cell_loc = (int(meter_coords[0]/self.ds[1]),int(meter_coords[1]/self.ds[2]))
		elif self.perpendicular_axis == 'y':
			cell_loc = (int(meter_coords[0]/self.ds[0]),int(meter_coords[1]/self.ds[2]))
		elif self.perpendicular_axis == 'z':
			cell_loc = (int(meter_coords[0]/self.ds[0]),int(meter_coords[1]/self.ds[1]))

		return self.get_history_at_cell(component, cell_loc)

	def get_history_at_cell(self,component,cell_loc: tuple):
		
		component_history_list = [(t,self.get_component_at_cell(component,cell_loc,t)) for t in self.times]

		return np.asarray(component_history_list)
	
	def get_snapshot_all_cells(self,component, time_t, include_unsat = False):

		h5_data = self.h5_data
		t_group = self.get_time_t_group(time_t)
			
		if t_group is not None:
			full_set = np.array(h5_data[t_group][component])


			inds = self.cross_section_cells
			
			cross_set = full_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]

			flat_shape = [i for i in np.shape(cross_set) if i != 1]	 

			flattened_cross_set = np.reshape(cross_set,flat_shape)
			if len(np.shape(flattened_cross_set)) == 2:
				oriented_set = np.fliplr(flattened_cross_set).T
			else:
				oriented_set = np.flip(np.reshape(flattened_cross_set,(len(flattened_cross_set),1)))
			
			matid_set = np.array(h5_data[t_group]['Material_ID'])
			matid_cross_set = matid_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
			matid_flattened_cross_set = np.reshape(matid_cross_set,flat_shape)
			matid_oriented_set = np.fliplr(matid_flattened_cross_set).T
			plot_set = np.ma.masked_where(matid_oriented_set == 0, oriented_set)
			
			if include_unsat is False:
				sat_full_set = np.array(h5_data[t_group]['Liquid_Saturation'])
				sat_cross_set = sat_full_set[inds[0][0]:inds[0][1],inds[1][0]:inds[1][1],inds[2][0]:inds[2][1]]
				sat_flattened_cross_set = np.reshape(sat_cross_set,flat_shape)
				sat_oriented_set = np.fliplr(sat_flattened_cross_set).T
				sat_oriented_set[np.where(matid_oriented_set == 0)]=1
				sat_oriented_set[np.where(sat_oriented_set != 1)] = 0
				#unsat_oriented_set = np.ma.masked_where(sat_oriented_set == 1, sat_oriented_set)
				plot_set = np.ma.masked_where((sat_oriented_set == 0) | (matid_oriented_set == 0), oriented_set)
				
			return plot_set

	def plot_history_at_cell(self,component,cell_loc: tuple,):

		component_history_array = self.get_history_at_cell(component,cell_loc)
		
		ixs = component_history_array[:,0]
		jys = component_history_array[:,1]
		fig, ax = plt.subplots()

		ax.plot(ixs, jys)
		
		plt.show()

	def plot_history_at_m_coords(self,component: str, meter_coords: tuple, ax = None):

		component_history_array = self.get_history_at_m_coords(component,meter_coords)
		
		ixs = component_history_array[:,0]
		jys = component_history_array[:,1]

		if ax == None:
			fig, ax = plt.subplots()

			ax.plot(ixs, jys)
			
			plt.show()
		
		else:

			ax.plot(ixs,jys)

			return ax 