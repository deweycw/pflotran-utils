"""
PFLOTRAN Processing Module

A unified class for PFLOTRAN simulation data processing, visualization, and grid building.

Extends pflotranutils.CrossSection with:
- Project-specific path resolution
- Meander site configurations (MZT/MCP)
- Thermodynamic calculations
- Grid building utilities
"""

import math
import numpy as np
import pandas as pd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from pathlib import Path
from typing import Optional, List, Tuple, Any

from .h5_output.calc.cross_section import CrossSection

# Project root directory — set by caller via project_root parameter or class attribute
_PROJECT_ROOT = None

# Default search paths for data files (relative to project root)
_DATA_SEARCH_PATHS = [
    'pflotran/simulations',
    'pflotran/model-output',
    'data',
    'results',
    '.',
]


def _resolve_path(relative_path: str, search_paths: Optional[List[str]] = None,
                   project_root: Optional[Path] = None) -> Path:
    """
    Resolve a relative path by searching in standard project directories.

    Args:
        relative_path: Relative path to file (e.g., 'mzt19/pflotran-mzt19.h5')
        search_paths: Optional list of paths to search (relative to project root)
        project_root: Optional project root directory. If None, uses module-level _PROJECT_ROOT.

    Returns:
        Resolved absolute Path

    Raises:
        FileNotFoundError: If file not found in any search path
    """
    # If already absolute, return as-is
    path = Path(relative_path)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"File not found: {relative_path}")

    root = project_root or _PROJECT_ROOT
    if root is None:
        raise FileNotFoundError(
            f"Cannot resolve relative path '{relative_path}': no project_root set. "
            f"Pass project_root to PflotranProcessor or use an absolute path."
        )

    # Search in standard locations
    search_paths = search_paths or _DATA_SEARCH_PATHS

    for search_dir in search_paths:
        candidate = root / search_dir / relative_path
        if candidate.exists():
            return candidate

    # Also try directly from project root
    candidate = root / relative_path
    if candidate.exists():
        return candidate

    # Build helpful error message
    searched = [str(root / sp / relative_path) for sp in search_paths]
    raise FileNotFoundError(
        f"File not found: '{relative_path}'\n"
        f"Searched in:\n  " + "\n  ".join(searched)
    )


class PflotranProcessor(CrossSection):
    """
    PFLOTRAN results processor extending pflotranutils.CrossSection.

    Adds project-specific features:
    - Automatic path resolution for HDF5 files
    - Meander site configurations (MZT/MCP)
    - Thermodynamic calculations
    - Time series plotting with observation overlays

    Inherited from CrossSection/HDF5Output:
    - get_times(), get_components(), print_components()
    - plot_at_time(), plot_velocity_at_time()
    - get_history_at_cell(), get_history_at_m_coords()
    - get_snapshot_all_cells(), get_material_ids()

    Paths can be specified as:
    - Absolute paths: '/full/path/to/file.h5'
    - Relative paths: 'mzt19/pflotran-mzt19.h5' (auto-searched in data directories)

    Default search locations (relative to project root):
    - pflotran/simulations/
    - pflotran/model-output/
    - data/
    - results/
    """

    # Site-specific configurations
    MEANDER_CONFIG = {
        'MZ': {
            'distances': [1.0, 16, 27, 40, 50],
            'depths': [2.0, 2.0, 2.0, 2.0, 2.0],
            'loc_dist': {1: '1', 16: '2', 27: '3', 40: '4', 50: '5'},
            'obs_locs': ['MZT1-1D', 'MZT1-2D', 'MZT1-3D', 'MZT1-4D', 'MZT1-5D'],
            'loc_name': {'MZT1-1D': '1', 'MZT1-2D': '2', 'MZT1-3D': '3', 'MZT1-4D': '4', 'MZT1-5D': '5',
                        'mzt11': '1', 'mzt13': '3', 'mzt15': '5'}
        },
        'MC': {
            'distances': [0.5, 16, 31, 46, 60.0],
            #'depths': [1.7, 2.0, 2.1, 2.4, 2.5],
            'depths': [1.7,1.8,1.8,1.9,2.0],
            'loc_dist': {0.5: '1', 16: '2', 31: '3', 46: '4', 60.0: '5'},
            'obs_locs': ['MCP1-1D', 'MCP1-2D', 'MCP1-3D', 'MCP1-4D', 'MCP1-5D'],
            'loc_name': {'MCP1-1D': '1', 'MCP1-2D': '2', 'MCP1-3D': '3', 'MCP1-4D': '4', 'MCP1-5D': '5'}
        }
    }

    # Standard color scheme
    LOC_SYMBOLS = {'river': 's', '1': 'o', '2': 'p', '3': 'd', '4': 'P', '5': 'X'}

    # Class-level project root for reference (set via __init__ or class attribute)
    PROJECT_ROOT = _PROJECT_ROOT

    def __init__(self, h5_path: Optional[str] = None, meander: str = 'MZ',
                 perpendicular_axis: str = 'x', perp_loc: float = 0.0,
                 project_root: Optional[Path] = None):
        """
        Initialize the processor.

        Args:
            h5_path: Path to the HDF5 results file. Can be:
                     - Absolute path: '/full/path/to/file.h5'
                     - Relative path: 'mzt19/pflotran-mzt19.h5' (searched in data dirs)
            meander: Site identifier ('MZ' or 'MC')
            perpendicular_axis: Axis perpendicular to cross-section ('x', 'y', or 'z').
                               Default 'x' since these simulations have NX=1.
            perp_loc: Location along perpendicular axis in meters (default: 0.0)
            project_root: Optional project root directory for resolving relative paths.
                         If None, uses the class-level PROJECT_ROOT attribute.
        """
        if meander not in self.MEANDER_CONFIG:
            raise ValueError(f"Invalid meander: {meander}. Must be 'MZ' or 'MC'.")

        # Set instance-level project root (falls back to class attribute)
        self._project_root = Path(project_root) if project_root is not None else self.PROJECT_ROOT

        self.meander = meander
        self.config = self.MEANDER_CONFIG[meander]
        self._h5_path_input = h5_path
        self._h5_path_resolved: Optional[Path] = None

        # Resolve path if provided
        if h5_path is not None:
            self._h5_path_resolved = _resolve_path(h5_path, project_root=self._project_root)
            # Initialize parent CrossSection with resolved path
            super().__init__(str(self._h5_path_resolved), perpendicular_axis)
            # Initialize cross-section cells for get_snapshot_all_cells, etc.
            self.get_cells(perp_loc=perp_loc)

        self._setup_colors()

    @property
    def h5_path(self) -> Optional[Path]:
        """Get the resolved HDF5 file path."""
        return self._h5_path_resolved

    @h5_path.setter
    def h5_path(self, value: Optional[str]):
        """Set and resolve the HDF5 file path."""
        self._h5_path_input = value
        if value is not None:
            self._h5_path_resolved = _resolve_path(value, project_root=self._project_root)
        else:
            self._h5_path_resolved = None

    def _get_h5_path(self, h5_path: Optional[str] = None) -> Path:
        """
        Get the resolved HDF5 path, either from argument or instance.

        Args:
            h5_path: Optional path to HDF5 file

        Returns:
            Resolved Path object

        Raises:
            ValueError: If no path provided and no instance path set
        """
        if h5_path is not None:
            return _resolve_path(h5_path, project_root=self._project_root)
        if self._h5_path_resolved is not None:
            return self._h5_path_resolved
        raise ValueError("No HDF5 path provided and no instance path set")

    @staticmethod
    def _get_dataset(file: h5py.File, group_name: str, component: str) -> np.ndarray:
        """
        Safely get dataset from HDF5 file group.

        Args:
            file: Open h5py.File object
            group_name: Name of the time group
            component: Component/dataset name to retrieve

        Returns:
            Numpy array of the dataset

        Raises:
            KeyError: If group or component not found
        """
        group = file[group_name]
        if not isinstance(group, h5py.Group):
            raise KeyError(f"'{group_name}' is not a group")
        dataset = group[component]
        return np.array(dataset)

    def _setup_colors(self):
        """Setup color scheme for plotting."""
        cmap = mpl.colormaps['viridis']
        self.cmaplist = [cmap(i) for i in np.arange(0, 1, 0.2)]
        self.loc_colors = {
            'river': 'mediumblue',
            '1': self.cmaplist[0],
            '2': self.cmaplist[1],
            '3': self.cmaplist[2],
            '4': self.cmaplist[3],
            '5': self.cmaplist[4]
        }

    def _get_time_unit(self) -> str:
        """
        Detect the time unit from the HDF5 file by examining the time group keys.

        Returns:
            Time unit string: 'h' (hours), 'y' (years), 'd' (days), 's' (seconds)
        """
        if self._h5_path_resolved is None:
            return 'h'  # Default to hours

        try:
            with h5py.File(str(self._h5_path_resolved), 'r') as f:
                for key in f.keys():
                    if key.startswith('Time:'):
                        # Extract unit from key like 'Time:  1.00000E+00 h'
                        parts = key.strip().split()
                        if parts:
                            unit = parts[-1]
                            if unit in ['h', 'y', 'd', 's', 'mo', 'w']:
                                return unit
        except Exception:
            pass

        return 'h'  # Default to hours

    def _convert_times_to_days(self, times: np.ndarray) -> np.ndarray:
        """
        Convert times array to days based on the detected time unit.

        Args:
            times: Array of time values in the file's native unit

        Returns:
            Array of time values in days
        """
        unit = self._get_time_unit()

        # Conversion factors to days
        conversions = {
            'h': 1.0 / 24.0,      # hours to days
            'd': 1.0,              # days (no conversion)
            'y': 365.25,           # years to days
            's': 1.0 / 86400.0,    # seconds to days
            'mo': 30.4375,         # months to days (approximate)
            'w': 7.0               # weeks to days
        }

        factor = conversions.get(unit, 1.0 / 24.0)  # Default to hours
        return times * factor

    # =========================================================================
    # Data Extraction Methods
    # =========================================================================

    def get_locs_year(self, data_dir: str) -> Tuple[List[Tuple[float, float]], Optional[int]]:
        """
        Get location tuples and year from data directory path.

        Args:
            data_dir: Path to data directory

        Returns:
            Tuple of (locations list, year or None)
        """
        dir_name = str.split(data_dir, '/')[-2]

        if 'mzt' in dir_name:
            distances = [1.0, 16, 27, 40, 50]
            depths = [2.0, 2.0, 2.0, 2.0, 2.0]
        else:
            distances = [0.5, 16, 31, 46, 60.]
            depths = [1.7, 2.0, 2.1, 2.4, 2.5]

        locs = [(i, d) for i, d in zip(distances, depths)]

        if '18' in dir_name:
            year = 2018
        elif '19' in dir_name:
            year = 2019
        else:
            year = None

        return locs, year

    def get_histories(self, depths: Optional[List[float]] = None,
                      components: Optional[List[str]] = None) -> Tuple[dict, np.ndarray]:
        """
        Get time series histories at all standard monitoring locations.

        Extracts component histories at all configured distances for the meander site.
        Returns a nested dictionary keyed by distance then component name.

        Args:
            depths: Optional list of depths for each location. If None, uses
                    the configured depths from MEANDER_CONFIG for the site.
            components: Optional list of components to extract. If None, extracts all.

        Returns:
            Tuple of (results_dict, times_array) where:
            - results_dict: {distance: {component: values_array, ...}, ...}
            - times_array: Array of time values

        Example:
            results, times = processor.get_histories()
            fe2_at_16m = results[16]['Free_Fe++ [M]']
        """
        distances = self.config['distances']

        if depths is None:
            # Use configured depths for each observation location
            depths = self.config.get('depths', [1.0] * len(distances))

        if len(depths) != len(distances):
            raise ValueError(f"depths list length ({len(depths)}) must match "
                           f"distances length ({len(distances)})")

        locs = [(d, z) for d, z in zip(distances, depths)]

        if components is None:
            if not hasattr(self, 'component_list') or self.component_list is None:
                raise ValueError("No components available. Ensure HDF5 file was loaded.")
            components = list(self.component_list)

        if not hasattr(self, 'times') or self.times is None:
            raise ValueError("No times available. Ensure HDF5 file was loaded.")
        times = self.times

        results: dict = {d: {} for d in distances}

        for component in components:
            for i, loc in enumerate(locs):
                dset = self.get_history_at_m_coords(component=component, meter_coords=loc)
                c_dset = dset[:, 1]
                distance = distances[i]
                results[distance][component] = c_dset

        return results, times

    def extract_component_list(self, h5_path: Optional[str] = None) -> List[str]:
        """
        Extract list of available components from HDF5 file.

        Args:
            h5_path: Path to HDF5 file (uses instance path if not provided)

        Returns:
            List of component names

        Note:
            If using instance path and inheriting from CrossSection,
            components are also available via self.component_list after init.
        """
        # If using instance path and already initialized, use inherited attribute
        if h5_path is None and hasattr(self, 'component_list'):
            return list(self.component_list)

        resolved_path = self._get_h5_path(h5_path)
        with h5py.File(resolved_path, 'r') as f:
            groups = list(f.keys())
            for gg in groups:
                if 'Time' in gg:
                    group = f[gg]
                    if isinstance(group, h5py.Group):
                        return list(group.keys())
        return []

    def extract_times(self, h5_path: Optional[str] = None) -> np.ndarray:
        """
        Extract time steps from HDF5 file.

        Args:
            h5_path: Path to HDF5 file (uses instance path if not provided)

        Returns:
            Array of time values (column vector)

        Note:
            If using instance path and inheriting from CrossSection,
            times are also available via self.times after init.
        """
        # If using instance path and already initialized, use inherited attribute
        if h5_path is None and hasattr(self, 'times'):
            return np.array([self.times]).T

        resolved_path = self._get_h5_path(h5_path)
        with h5py.File(resolved_path, 'r') as f:
            groups = list(f.keys())
            times = []
            for gg in groups:
                if 'Time' in gg:
                    times.append(float(gg[7:18]))
        return np.array([times]).T

    def extract_component_transect(self, component: str, return_times: bool = False,
                                   rm_bc_cells: bool = False, h5_path: Optional[str] = None) -> Any:
        """
        Extract component data along transect for all times.

        Args:
            component: Component name to extract
            return_times: Whether to return times array
            rm_bc_cells: Whether to remove boundary condition cells
            h5_path: Path to HDF5 file

        Returns:
            Data array, or tuple of (times, data) if return_times=True
        """
        resolved_path = self._get_h5_path(h5_path)
        with h5py.File(resolved_path, 'r') as f:
            groups = list(f.keys())
            dx, dy = 1, 1
            nx, ny = int(1/dx), int(1/dy)

            component_list = []
            times = []

            for gg in groups:
                if 'Time' in gg:
                    times.append(float(gg[7:18]))
                    dset = self._get_dataset(f, gg, component)
                    if rm_bc_cells:
                        component_transect = dset[nx, ny, 1:-1]
                    else:
                        component_transect = dset[nx-1, ny-1, :]
                    component_list.append(component_transect)

            component_list = np.asarray(component_list)
            times_arr = np.array([times]).T
            data = np.append(times_arr, component_list, axis=1)
            data = data[data[:, 0].argsort()]
            times_sorted = data[:, 0]
            data = data[:, 1:]

            if return_times:
                return times_sorted, data
            return data

    def extract_component_at_time(self, component: str, dz: float, dist: float,
                                  time: float, h5_path: Optional[str] = None) -> Optional[float]:
        """
        Extract component value at specific location and time.

        Args:
            component: Component name
            dz: Vertical discretization
            dist: Distance along transect
            time: Time step
            h5_path: Path to HDF5 file

        Returns:
            Component value at specified coordinates, or None if not found
        """
        resolved_path = self._get_h5_path(h5_path)
        with h5py.File(resolved_path, 'r') as f:
            groups = list(f.keys())
            dx, dy = 1, 1
            nx, ny, nz = int(1/dx), int(1/dy), int(dist/dz)

            time_str = str(np.format_float_scientific(time, precision=5, unique=False)).replace('e', 'E')
            for gg in groups:
                if time_str in gg:
                    dset = self._get_dataset(f, gg, component)
                    return float(dset[nx, ny, nz])
        return None

    def extract_transect_at_time(self, component: str, time: float,
                                 discretization_dir: str = 'z', rm_bc_cells: bool = False,
                                 h5_path: Optional[str] = None) -> np.ndarray:
        """
        Extract transect data at specific time.

        Args:
            component: Component name
            time: Time step
            discretization_dir: Direction of transect ('x', 'y', or 'z')
            rm_bc_cells: Whether to remove BC cells
            h5_path: Path to HDF5 file

        Returns:
            Component data along transect
        """
        resolved_path = self._get_h5_path(h5_path)
        with h5py.File(resolved_path, 'r') as f:
            groups = list(f.keys())
            component_list = []
            time_str = str(np.format_float_scientific(time, precision=5, unique=False)).replace('e', 'E')

            if discretization_dir == 'z':
                dx, dy = 1, 1
                nx, ny = int(1/dx), int(1/dy)
                for gg in groups:
                    if time_str in gg:
                        dset = self._get_dataset(f, gg, component)
                        if rm_bc_cells:
                            component_list.append(dset[nx-1, ny-1, 1:-1])
                        else:
                            component_list.append(dset[nx-1, ny-1, :])

            elif discretization_dir == 'x':
                dz, dy = 1, 1
                nz, ny = int(1/dz), int(1/dy)
                for gg in groups:
                    if time_str in gg:
                        dset = self._get_dataset(f, gg, component)
                        if rm_bc_cells:
                            component_list.append(dset[1:-1, ny-1, nz-1])
                        else:
                            component_list.append(dset[:, ny-1, nz-1])

            elif discretization_dir == 'y':
                dz, dx = 1, 1
                nz, nx = int(1/dz), int(1/dx)
                for gg in groups:
                    if time_str in gg:
                        dset = self._get_dataset(f, gg, component)
                        if rm_bc_cells:
                            component_list.append(dset[nx-1, 1:-1, nz-1])
                        else:
                            component_list.append(dset[nx-1, :, nz-1])

            return np.asarray(component_list)

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def _get_unit_factor(self, unit: Optional[str]) -> float:
        """Get conversion factor for unit."""
        if unit is None:
            return 1.0
        factors = {'uM': 1e6, 'mM': 1e3}
        return factors.get(unit, 1.0)

    def plot_profile(self, results: dict, component: str, time: float, d: float,
                     xyz: str, ax: Optional[Axes] = None, unit: Optional[str] = None,
                     flip: bool = False, logscale: bool = False) -> Axes:
        """
        Plot concentration profile at specific time.

        Args:
            results: Dictionary of results data
            component: Component to plot
            time: Time step
            d: Discretization
            xyz: Direction ('x', 'y', or 'z')
            ax: Matplotlib axes
            unit: Unit for display
            flip: Whether to flip axis
            logscale: Whether to use log scale

        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots()

        component_data = results[component][time].flatten()

        if logscale:
            component_data[component_data < 0] = np.log10(np.abs(component_data[component_data < 0]))
            component_data[component_data == 0] = 0

        length = len(component_data) * d
        factor = self._get_unit_factor(unit)

        if xyz == 'z':
            y = np.atleast_2d(np.arange(0, length, d))
            x = np.atleast_2d(component_data)
            ax.scatter([xp * factor for xp in x.flatten()], y.flatten(), label=component)
            if flip:
                ax.set_ylim(0, length)
            if unit == 'rate':
                label = 'Log10(Rate [mol_m^3-sec])' if logscale else 'Rate [mol_m^3-sec]'
                ax.set_xlabel(label)
            elif unit == 'pH':
                ax.set_xlabel('pH')
            else:
                ax.set_xlabel(f'Concentration [{unit}]')
            ax.set_ylabel('Distance [m]')
        else:
            x = np.atleast_2d(np.arange(0, length, d))
            y = np.atleast_2d(component_data)
            ax.scatter(x.flatten(), [yp * factor for yp in y.flatten()], label=component)
            if unit == 'rate':
                ax.set_ylabel('Rate [mol_m^3-sec]')
            elif unit == 'pH':
                ax.set_ylabel('pH')
            else:
                ax.set_ylabel(f'Concentration [{unit}]')
            ax.set_xlabel('Distance [m]')

        return ax

    def plot_time_series(self, component_data: Any, times: List, distance: float,
                         discretization: float = 0.5, ax: Optional[Axes] = None,
                         unit: Optional[str] = None, startdate: Any = None,
                         reverse: bool = False) -> Axes:
        """
        Plot time series at specific distance.

        Args:
            component_data: Component data array (already extracted for the location)
            times: List of time values
            distance: Distance along transect (used for color mapping)
            discretization: Cell size (kept for API compatibility, not used)
            ax: Matplotlib axes
            unit: Unit for display
            startdate: Start date for x-axis
            reverse: Whether to reverse sign

        Returns:
            Matplotlib axes
        """
        _ = discretization  # Kept for API compatibility
        if ax is None:
            _, ax = plt.subplots()

        factor = self._get_unit_factor(unit)
        loc_dist = self.config['loc_dist']

        y = []
        times = [int(t) for t in times]
        times.sort()

        for t in range(len(times)):
            ct = component_data[t]
            y.append(ct * -1 if reverse else ct)

        color = self.loc_colors[loc_dist.get(distance, '1')]

        if startdate:
            times_delta = [np.timedelta64(int(t), 'h') for t in times]
            datex = [(startdate + dt) for dt in times_delta]
            ax.plot(datex, [yp * factor for yp in y], color=color)
            ax.set_xlabel('Date')
        else:
            ax.plot(times, [yp * factor for yp in y], color=color)
            ax.set_xlabel('Time [h]')

        if unit == 'pH':
            ax.set_ylabel('pH')
        else:
            ax.set_ylabel(f'Concentration [{unit}]')

        return ax

    def plot_pressure_time_series(self, component_data: dict, distance: float,
                                  discretization: float, ax: Optional[Axes] = None,
                                  unit: Optional[str] = None, startdate: Any = None) -> Axes:
        """
        Plot pressure time series.

        Args:
            component_data: Dictionary of pressure data by time
            distance: Distance along transect
            discretization: Cell size
            ax: Matplotlib axes
            unit: Unit for display ('mPa' or 'mASL')
            startdate: Start date for x-axis

        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots()

        factors = {'mPa': 1e3, 'mASL': (9.81 * 998)}
        factor = factors.get(unit, 1.0) if unit else 1.0

        loc_dist = {2.0: '1', 16: '2', 27: '3', 40: '4', 50: '5'}

        y = []
        idx = int(distance / discretization) - 1
        times = list(component_data.keys())
        times.sort()

        for t in times:
            ct = component_data[t][0][idx]
            y.append(ct - 101325)

        color = self.loc_colors[loc_dist.get(distance, '1')]

        if startdate:
            times_delta = [np.timedelta64(int(t), 'h') for t in times]
            datex = [(startdate + dt) for dt in times_delta]

            if unit == 'mASL':
                ax.plot(datex, [(yp / factor) + 2717.5 for yp in y],
                       alpha=1.0, linestyle='--', linewidth=0.75, color=color)
                ax.set_ylabel(f'Elevation [{unit}]')
            else:
                ax.plot(datex, [yp / factor for yp in y],
                       alpha=1.0, linestyle='--', linewidth=0.75)
                ax.set_ylabel(f'Pressure [{unit}]')
            ax.set_xlabel('Date')
        else:
            ax.plot(times, [yp / factor for yp in y], linestyle='--')
            ax.set_xlabel('Time [h]')
            ax.set_ylabel(f'Pressure [{unit}]')

        return ax

    def plot_observations(self, df: Any, location: str, ax: Optional[Axes] = None,
                         color: Optional[str] = None) -> Axes:
        """
        Plot observation data.

        Args:
            df: DataFrame with observations
            location: Location identifier
            ax: Matplotlib axes
            color: Color for markers

        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots()

        times = df['Date and time']
        kwargs = {'marker': 'x', 's': 1, 'alpha': 0.5}
        if color:
            kwargs['color'] = color

        ax.scatter(times, df[location], **kwargs)
        ax.set_xlabel('Date')
        ax.set_ylabel('Elevation (mASL)')

        return ax

    # =========================================================================
    # Chemical Component Plotting (convenience methods)
    # =========================================================================

    # Mapping from simulation component names to observation column names
    COMPONENT_TO_OBS_MAP = {
        'Total_Fe++ [M]': 'Fe',
        'Free_Fe++ [M]': 'Fe',
        'Total_SO4-- [M]': 'SO4',
        'Free_SO4-- [M]': 'SO4',
        'Total_Ca++ [M]': 'Ca',
        'Free_Ca++ [M]': 'Ca',
        'Total_Mg++ [M]': 'Mg',
        'Free_Mg++ [M]': 'Mg',
        'Total_Na+ [M]': 'Na',
        'Free_Na+ [M]': 'Na',
        'Total_K+ [M]': 'K',
        'Free_K+ [M]': 'K',
        'Total_Cl- [M]': 'Cl',
        'Free_Cl- [M]': 'Cl',
        'Total_HCO3- [M]': 'TIC',
        'Free_HCO3- [M]': 'TIC',
        'Total_NH3(aq) [M]': 'NH4',
        'Free_NH3(aq) [M]': 'NH4',
        'Total_NO3- [M]': 'NO3',
        'Free_NO3- [M]': 'NO3',
        'Total_O2(aq) [M]': 'DO',
        'Free_O2(aq) [M]': 'DO',
        'Total_SiO2(aq) [M]': 'Si',
        'Free_SiO2(aq) [M]': 'Si',
        'Total_Mn++ [M]': 'Mn',
        'Free_Mn++ [M]': 'Mn',
        'pH': 'pH',
        'Total_DOC [M]': 'NPOC',
        'Total_SOC(aq) [M]': 'NPOC',
        'Free_SOC(aq) [M]': 'NPOC',
    }

    # Components in mM in observation data
    OBS_MM_COMPONENTS = ['TC', 'NPOC', 'TIC', 'NH4', 'Cl', 'NO2', 'SO4', 'NO3']

    # Components in M in observation data
    OBS_M_COMPONENTS = ['Na', 'Na_sd', 'Mg', 'Mg_sd', 'Si', 'Si_sd', 'Si-1', 'Si-1_sd',
                        'K', 'K_sd', 'Mn', 'Mn_sd', 'Ca', 'Ca_sd', 'Fe', 'Fe_sd',
                        'Ni', 'Ni_sd', 'Cu', 'Cu_sd', 'Zn', 'Zn_sd', 'Sr', 'Sr_sd',
                        'Ba', 'Ba_sd', 'U', 'U_sd', 'PO', 'PO_sd']

    # Components in mg/L in observation data
    OBS_MGL_COMPONENTS = ['DO']

    @staticmethod
    def _get_default_unit(obs_component_name: str) -> str:
        """
        Get the default unit based on the observation component's native units.

        Args:
            obs_component_name: Name of the observation component

        Returns:
            Default unit for the component
        """
        obs_mM_components = ['TC', 'NPOC', 'TIC', 'NH4', 'Cl', 'NO2', 'SO4', 'NO3']
        obs_M_components = ['Na', 'Na_sd', 'Mg', 'Mg_sd', 'Si', 'Si_sd', 'Si-1', 'Si-1_sd',
                           'K', 'K_sd', 'Mn', 'Mn_sd', 'Ca', 'Ca_sd', 'Fe', 'Fe_sd',
                           'Ni', 'Ni_sd', 'Cu', 'Cu_sd', 'Zn', 'Zn_sd', 'Sr', 'Sr_sd',
                           'Ba', 'Ba_sd', 'U', 'U_sd', 'PO', 'PO_sd']

        if obs_component_name == 'pH':
            return 'pH'
        elif obs_component_name == 'DO':
            return 'uM'  # Show DO in μM (converted from mg/L)
        elif obs_component_name in obs_mM_components:
            return 'mM'
        elif obs_component_name in obs_M_components:
            return 'M'
        else:
            return 'mM'  # Default fallback

    @staticmethod
    def _get_obs_unit_factor(obs_component_name: str, target_unit: str) -> float:
        """
        Get the conversion factor for observation data based on its native units.

        Args:
            obs_component_name: Name of the observation component
            target_unit: Target unit for display (e.g., 'mM', 'uM', 'M')

        Returns:
            Conversion factor to apply to observation data
        """
        obs_mM_components = ['TC', 'NPOC', 'TIC', 'NH4', 'Cl', 'NO2', 'SO4', 'NO3']
        obs_M_components = ['Na', 'Na_sd', 'Mg', 'Mg_sd', 'Si', 'Si_sd', 'Si-1', 'Si-1_sd',
                           'K', 'K_sd', 'Mn', 'Mn_sd', 'Ca', 'Ca_sd', 'Fe', 'Fe_sd',
                           'Ni', 'Ni_sd', 'Cu', 'Cu_sd', 'Zn', 'Zn_sd', 'Sr', 'Sr_sd',
                           'Ba', 'Ba_sd', 'U', 'U_sd', 'PO', 'PO_sd']
        obs_mgL_components = ['DO']

        # Determine native unit of observation data
        if obs_component_name in obs_mM_components:
            obs_native_unit = 'mM'
        elif obs_component_name in obs_M_components:
            obs_native_unit = 'M'
        elif obs_component_name in obs_mgL_components:
            obs_native_unit = 'mg/L'
        else:
            obs_native_unit = 'M'  # Default assumption

        # Handle mg/L conversions for dissolved oxygen (O2)
        if obs_native_unit == 'mg/L' and obs_component_name == 'DO':
            MW_O2 = 32.0  # g/mol
            if target_unit == 'M':
                return 1.0 / (MW_O2 * 1000)  # mg/L to M
            elif target_unit == 'mM':
                return 1.0 / MW_O2  # mg/L to mM
            elif target_unit == 'uM':
                return 1000.0 / MW_O2  # mg/L to uM
            else:
                return 1.0

        # Handle pH - always unitless
        if obs_component_name == 'pH' or target_unit == 'pH':
            return 1.0

        # Standard unit conversions
        if obs_native_unit == 'M' and target_unit == 'mM':
            return 1e3
        elif obs_native_unit == 'M' and target_unit == 'uM':
            return 1e6
        elif obs_native_unit == 'mM' and target_unit == 'M':
            return 1e-3
        elif obs_native_unit == 'mM' and target_unit == 'uM':
            return 1e3
        elif obs_native_unit == 'uM' and target_unit == 'M':
            return 1e-6
        elif obs_native_unit == 'uM' and target_unit == 'mM':
            return 1e-3
        else:
            return 1.0  # Same units or unknown

    @staticmethod
    def _get_sim_unit_factor(target_unit: str) -> float:
        """
        Get the conversion factor for simulation data (always in M) to target unit.

        Args:
            target_unit: Target unit for display

        Returns:
            Conversion factor to apply to simulation data
        """
        if target_unit == 'mM':
            return 1e3
        elif target_unit == 'uM':
            return 1e6
        elif target_unit == 'pH':
            return 1.0
        else:
            return 1.0  # Keep in M

    def plot_component_histories(self, component_name: str, ax: Axes,
                                  startdate: Optional[Any] = None,
                                  results: Optional[dict] = None, times: Optional[List] = None,
                                  distances: Optional[List] = None,
                                  chem_obs: Any = None, unit: Optional[str] = None,
                                  obs_component_name: Optional[str] = None,
                                  reverse: bool = False,
                                  plot_obs_average: bool = False,
                                  depths: Optional[List[float]] = None) -> Axes:
        """
        Plot time series histories of a component at all monitoring locations.

        Plots the component history at each configured distance (monitoring well location).
        If results/times are not provided, automatically extracts data using get_histories().

        Args:
            component_name: Name of the component to plot (e.g., 'Total_Ca++ [M]')
            ax: Axes object to plot on
            startdate: Optional starting date for time series (e.g., np.datetime64('2019-05-01')).
                      If None, plots with x-axis in days starting at day=0.
            results: Optional dict of results by distance. If None, calls get_histories().
            times: Optional time points. If None, uses self.times.
            distances: Optional distance points. If None, uses self.config['distances'].
            chem_obs: Optional chemical observations DataFrame. With startdate, plots time series.
                      Without startdate, plots mean with stdev shading.
            unit: Unit for y-axis display. If None, defaults to observation units.
                  For pH, always treated as unitless regardless of input.
            obs_component_name: Name of component in observations data (if different).
                               If None, will try to infer from component_name.
            reverse: Whether to reverse the sign of the simulation data
            plot_obs_average: Whether to plot average observation values as horizontal lines
            depths: Optional depths for get_histories() if extracting data

        Returns:
            Updated axes object

        Example:
            # Simple usage with dates - auto-extracts data
            processor.plot_component_histories('Total_Fe++ [M]', ax, startdate=startdate)

            # Simple usage without dates - x-axis in days
            processor.plot_component_histories('Total_Fe++ [M]', ax)

            # With observations (requires startdate)
            processor.plot_component_histories('Total_Fe++ [M]', ax, startdate=startdate, chem_obs=obs_df)

            # Pre-extracted data (faster for multiple plots)
            results, times = processor.get_histories()
            processor.plot_component_histories('Total_Fe++ [M]', ax, results=results)
        """
        # Use defaults from processor if not provided
        if distances is None:
            distances = list(self.config['distances'])

        # Extract results if not provided
        if results is None:
            results, extracted_times = self.get_histories(depths=depths, components=[component_name])
            times_array = extracted_times
        else:
            times_array = self.times if times is None else np.array(times)

        # Validate inputs
        if distances is None or len(distances) == 0:
            print("Warning: No distances provided!")
            return ax

        if times_array is None:
            print("Warning: No times available!")
            return ax

        # Check component availability
        first_distance = distances[0]
        if results is None or first_distance not in results:
            print(f"Warning: Distance {first_distance} not found in results!")
            return ax

        available_components = list(results[first_distance].keys())
        if component_name not in available_components:
            print(f"Warning: Component '{component_name}' not found in results!")
            print(f"Available components: {available_components[:10]}...")
            return ax

        # Handle pH specially - no unit conversion
        is_pH = 'pH' in component_name

        # Infer observation component name if not provided
        if obs_component_name is None:
            obs_component_name = self.COMPONENT_TO_OBS_MAP.get(component_name)
            if obs_component_name is None and chem_obs is not None:
                print(f"Warning: Could not infer observation name for '{component_name}'")

        # Determine unit and conversion factors
        if is_pH:
            target_unit = 'pH'
            sim_unit_factor = 1.0
            obs_unit_factor = 1.0
            ylabel = 'pH'
        else:
            # Get default unit if not specified
            if unit is None and obs_component_name is not None:
                target_unit = self._get_default_unit(obs_component_name)
            else:
                target_unit = unit if unit else 'M'

            # Simulation data is always in M, convert to target unit
            sim_unit_factor = self._get_sim_unit_factor(target_unit)

            # Observation data needs conversion based on its native units
            if obs_component_name is not None:
                obs_unit_factor = self._get_obs_unit_factor(obs_component_name, target_unit)
            else:
                obs_unit_factor = 1.0

            ylabel = f'Concentration [{target_unit}]' if target_unit else 'Concentration [M]'

        # Plot simulation results for each distance
        loc_dist = self.config['loc_dist']
        for distance in distances:
            if distance not in results:
                print(f"Warning: Distance {distance} not found in results, skipping...")
                continue

            component_data = results[distance][component_name]

            # Apply reverse if needed
            if reverse:
                component_data = -1 * np.array(component_data)

            # Apply unit conversion (simulation data is in M)
            y_data = np.array(component_data) * sim_unit_factor

            # Create time axis
            if startdate:
                # Convert times to days first, then to timedelta
                times_in_days = self._convert_times_to_days(np.array(times_array))
                times_delta = [np.timedelta64(int(t * 24), 'h') for t in times_in_days]
                x_data = [(startdate + dt) for dt in times_delta]
            else:
                # No startdate: convert to days starting at 0
                x_data = list(self._convert_times_to_days(np.array(times_array)))

            # Get color for this distance
            loc_key = loc_dist.get(distance, '1')
            color = self.loc_colors.get(loc_key, 'gray')

            ax.plot(x_data, y_data, color=color, label=f'{distance}m')

        # Plot observations if provided
        if chem_obs is not None and obs_component_name is not None:
            try:
                if obs_component_name in chem_obs.columns or 'Well' in chem_obs.columns:
                    for loc in self.config['obs_locs']:
                        df = chem_obs[chem_obs['Well'] == loc].copy()
                        if obs_component_name not in df.columns:
                            continue

                        mask = df[obs_component_name].isna()
                        df = df[~mask]

                        if len(df) == 0:
                            continue

                        loc_id = self.config['loc_name'].get(loc, '1')
                        color = self.loc_colors.get(loc_id, 'gray')
                        marker = self.LOC_SYMBOLS.get(loc_id, 'o')

                        # Apply unit conversion to observation data
                        obs_values = df[obs_component_name] * obs_unit_factor

                        if startdate:
                            # Plot time series of observations
                            ax.plot(df['Date'], obs_values,
                                   linestyle='-.', linewidth=0.5,
                                   color=color, marker=marker, markersize=3,
                                   alpha=0.7)

                            # Plot average as horizontal line if requested
                            if plot_obs_average:
                                avg_val = obs_values.mean()
                                ax.axhline(y=avg_val, color=color, linestyle=':',
                                          linewidth=0.5, alpha=0.5)
                        else:
                            # No startdate: plot average with stdev shading
                            avg_val = obs_values.mean()
                            std_val = obs_values.std()

                            # Get x-axis limits for horizontal line/shading
                            x_min = min(x_data) if x_data else 0
                            x_max = max(x_data) if x_data else 1

                            # Plot average as horizontal line
                            ax.axhline(y=avg_val, color=color, linestyle='--',
                                      linewidth=1.5, alpha=0.8, label=f'{loc} obs mean')

                            # Plot stdev as shaded region
                            ax.axhspan(avg_val - std_val, avg_val + std_val,
                                       color=color, alpha=0.15)

            except Exception as e:
                print(f"Warning: Could not plot observations: {e}")

        # Format axes
        if startdate:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d-%b"))
            ax.set_xlabel('Date')
        else:
            ax.set_xlabel('Days')
        ax.set_ylabel(ylabel)

        return ax

    def plot_validation(self, component_name: str, startdate: Any,
                        chem_obs: Any, ax: Axes,
                        results: Optional[dict] = None,
                        distances: Optional[List] = None,
                        unit: Optional[str] = None,
                        obs_component_name: Optional[str] = None,
                        depths: Optional[List[float]] = None,
                        show_stats: bool = True,
                        show_one_to_one: bool = True,
                        color_by_location: bool = True,
                        show_legend: bool = True) -> Axes:
        """
        Plot observed vs simulated values for model validation (1:1 plot).

        For each observation date, finds the corresponding simulated value
        and plots observed (x-axis) vs simulated (y-axis).

        Args:
            component_name: Name of the component to plot (e.g., 'Total_Fe++ [M]')
            startdate: Starting date for simulation (e.g., np.datetime64('2019-05-01'))
            chem_obs: Chemical observations DataFrame with 'Date' and 'Well' columns
            ax: Axes object to plot on
            results: Optional dict of results by distance. If None, calls get_histories().
            distances: Optional distance points. If None, uses self.config['distances'].
            unit: Unit for display. If None, auto-determines based on component.
            obs_component_name: Name of component in observations (if different).
            depths: Optional depths for get_histories() if extracting data.
            show_stats: Whether to display R², RMSE statistics on plot.
            show_one_to_one: Whether to show 1:1 reference line.
            color_by_location: Whether to color points by monitoring location.
            show_legend: Whether to show legend.

        Returns:
            Updated axes object

        Example:
            fig, ax = plt.subplots()
            processor.plot_validation(
                'Total_Fe++ [M]',
                startdate=np.datetime64('2019-05-01'),
                chem_obs=obs_df,
                ax=ax
            )
        """
        # Use defaults from processor if not provided
        if distances is None:
            distances = list(self.config['distances'])

        # Extract results if not provided
        if results is None:
            results, _ = self.get_histories(depths=depths, components=[component_name])

        # Infer observation component name if not provided
        if obs_component_name is None:
            obs_component_name = self.COMPONENT_TO_OBS_MAP.get(component_name)
            if obs_component_name is None:
                print(f"Warning: Could not infer observation name for '{component_name}'")
                return ax

        # Check if observation component exists
        if obs_component_name not in chem_obs.columns:
            print(f"Warning: '{obs_component_name}' not found in observation data")
            return ax

        # Determine unit and conversion factors
        is_pH = 'pH' in component_name
        if is_pH:
            target_unit = 'pH'
            sim_unit_factor = 1.0
            obs_unit_factor = 1.0
        else:
            if unit is None:
                target_unit = self._get_default_unit(obs_component_name)
            else:
                target_unit = unit
            sim_unit_factor = self._get_sim_unit_factor(target_unit)
            obs_unit_factor = self._get_obs_unit_factor(obs_component_name, target_unit)

        # Get simulation times as datetime
        sim_times = self.times
        sim_datetimes = [startdate + np.timedelta64(int(t), 'h') for t in sim_times]

        # Collect all observed vs simulated pairs
        all_observed = []
        all_simulated = []
        all_locations = []

        loc_dist = self.config['loc_dist']

        for loc in self.config['obs_locs']:
            # Get observations for this location
            df = chem_obs[chem_obs['Well'] == loc].copy()
            if obs_component_name not in df.columns:
                continue

            mask = df[obs_component_name].isna()
            df = df[~mask]

            if len(df) == 0:
                continue

            # Get the distance for this location
            loc_id = self.config['loc_name'].get(loc, '1')
            distance = None
            for d, lid in loc_dist.items():
                if lid == loc_id:
                    distance = d
                    break

            if distance is None or distance not in results:
                continue

            # Get simulation data for this distance
            sim_data = np.array(results[distance][component_name]) * sim_unit_factor

            # For each observation, find corresponding simulation value
            for _, row in df.iterrows():
                obs_date = row['Date']
                obs_value = row[obs_component_name] * obs_unit_factor

                # Find closest simulation time
                if hasattr(obs_date, 'to_numpy'):
                    obs_datetime = obs_date.to_numpy()
                else:
                    obs_datetime = np.datetime64(obs_date)

                # Calculate time differences
                time_diffs = [abs((obs_datetime - sd).astype('timedelta64[h]').astype(float))
                             for sd in sim_datetimes]
                closest_idx = np.argmin(time_diffs)

                # Only include if within 12 hours of a simulation output
                if time_diffs[closest_idx] <= 12:
                    sim_value = sim_data[closest_idx]
                    all_observed.append(obs_value)
                    all_simulated.append(sim_value)
                    all_locations.append(loc_id)

        if len(all_observed) == 0:
            print("Warning: No matching observation/simulation pairs found")
            return ax

        all_observed = np.array(all_observed)
        all_simulated = np.array(all_simulated)

        # Plot points
        if color_by_location:
            for loc_id in set(all_locations):
                mask = np.array([l == loc_id for l in all_locations])
                color = self.loc_colors.get(loc_id, 'gray')
                marker = self.LOC_SYMBOLS.get(loc_id, 'o')
                ax.scatter(all_observed[mask], all_simulated[mask],
                          color=color, marker=marker, s=30, alpha=0.7,
                          label=f'Loc {loc_id}')
        else:
            ax.scatter(all_observed, all_simulated, alpha=0.7, s=30)

        # Calculate axis limits (filter out NaN/Inf values)
        valid_obs = all_observed[np.isfinite(all_observed)]
        valid_sim = all_simulated[np.isfinite(all_simulated)]
        if len(valid_obs) > 0 and len(valid_sim) > 0:
            min_val = min(valid_obs.min(), valid_sim.min())
            max_val = max(valid_obs.max(), valid_sim.max())
        else:
            min_val, max_val = 0, 1  # Default fallback
        margin = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
        line_min = min_val - margin
        line_max = max_val + margin

        # Plot 1:1 line
        if show_one_to_one:
            ax.plot([line_min, line_max], [line_min, line_max], 'k--', linewidth=1, label='1:1 line')

        # Calculate and plot trendline (linear regression)
        from scipy import stats as scipy_stats
        reg_result = scipy_stats.linregress(all_observed, all_simulated)
        slope = float(reg_result.slope)
        intercept = float(reg_result.intercept)
        r_value = float(reg_result.rvalue)
        trend_x = np.array([line_min, line_max])
        trend_y = slope * trend_x + intercept
        ax.plot(trend_x, trend_y, 'r-', linewidth=1.5, alpha=0.8, label='Trend')

        # Set equal axis limits and ticks
        ax.set_xlim(line_min, line_max)
        ax.set_ylim(line_min, line_max)
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_xticks())
        ax.set_xlim(line_min, line_max)
        ax.set_ylim(line_min, line_max)

        # Calculate statistics
        r_squared = r_value ** 2
        rmse = np.sqrt(np.mean((all_simulated - all_observed) ** 2))
        bias = np.mean(all_simulated - all_observed)

        # Calculate NSE
        obs_mean = np.mean(all_observed)
        ss_res = np.sum((all_observed - all_simulated) ** 2)
        ss_tot = np.sum((all_observed - obs_mean) ** 2)
        nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Calculate KGE
        sim_mean = np.mean(all_simulated)
        obs_std = np.std(all_observed)
        sim_std = np.std(all_simulated)
        if obs_std > 0 and obs_mean != 0:
            alpha = sim_std / obs_std  # Variability ratio
            beta = sim_mean / obs_mean  # Bias ratio
            kge = 1 - np.sqrt((r_value - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge = np.nan

        # Print stats to console
        print(f"\n--- Validation Stats: {component_name} ---")
        print(f"  R²    = {r_squared:.3f}")
        print(f"  NSE   = {nse:.3f}")
        print(f"  KGE   = {kge:.3f}")
        print(f"  RMSE  = {rmse:.2e}")
        print(f"  Bias  = {bias:.2e}")
        print(f"  Slope = {slope:.3f}")
        print(f"  n     = {len(all_observed)}")

        # Display R² and RMSE on plot
        if show_stats:
            stats_text = f'R² = {r_squared:.3f}\nRMSE = {rmse:.2e}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Labels
        unit_label = f' [{target_unit}]' if target_unit and target_unit != 'pH' else ''
        ax.set_xlabel(f'Observed{unit_label}')
        ax.set_ylabel(f'Simulated{unit_label}')
        ax.set_title(f'{component_name}')

        if show_legend:
            ax.legend(loc='lower right', fontsize=8)

        ax.set_aspect('equal', adjustable='box')

        return ax

    def calculate_nse(self, component_name: str, startdate: Any,
                      chem_obs: Any,
                      results: Optional[dict] = None,
                      distances: Optional[List] = None,
                      obs_component_name: Optional[str] = None,
                      depths: Optional[List[float]] = None,
                      max_time_diff_hours: float = 12.0) -> dict:
        """
        Calculate Nash-Sutcliffe Efficiency (NSE) for a specified output parameter.

        NSE = 1 - (Σ(O_i - S_i)²) / (Σ(O_i - Ō)²)

        Where:
        - O_i = observed values
        - S_i = simulated values
        - Ō = mean of observed values

        NSE interpretation:
        - NSE = 1: Perfect match between simulated and observed
        - NSE = 0: Model is as accurate as using the mean of observations
        - NSE < 0: Mean of observations is a better predictor than the model

        Args:
            component_name: Name of the component (e.g., 'Total_Fe++ [M]')
            startdate: Starting date for simulation (e.g., np.datetime64('2019-05-01'))
            chem_obs: Chemical observations DataFrame with 'Date' and 'Well' columns
            results: Optional dict of results by distance. If None, calls get_histories().
            distances: Optional distance points. If None, uses self.config['distances'].
            obs_component_name: Name of component in observations (if different).
            depths: Optional depths for get_histories() if extracting data.
            max_time_diff_hours: Maximum time difference (hours) for matching
                                 observations to simulations. Default is 12 hours.

        Returns:
            Dictionary containing:
            - 'nse': Nash-Sutcliffe Efficiency value
            - 'nse_by_location': Dict of NSE values per monitoring location
            - 'n_pairs': Number of observation-simulation pairs used
            - 'observed': Array of observed values
            - 'simulated': Array of simulated values
            - 'rmse': Root Mean Square Error
            - 'bias': Mean bias (simulated - observed)

        Example:
            stats = processor.calculate_nse(
                'Total_Fe++ [M]',
                startdate=np.datetime64('2019-05-01'),
                chem_obs=obs_df
            )
            print(f"NSE: {stats['nse']:.3f}")
        """
        # Use defaults from processor if not provided
        if distances is None:
            distances = list(self.config['distances'])

        # Extract results if not provided
        if results is None:
            results, _ = self.get_histories(depths=depths, components=[component_name])

        # Infer observation component name if not provided
        if obs_component_name is None:
            obs_component_name = self.COMPONENT_TO_OBS_MAP.get(component_name)
            if obs_component_name is None:
                raise ValueError(f"Could not infer observation name for '{component_name}'. "
                               f"Please provide obs_component_name explicitly.")

        # Check if observation component exists
        if obs_component_name not in chem_obs.columns:
            raise ValueError(f"'{obs_component_name}' not found in observation data columns")

        # Determine unit conversion factors
        is_pH = 'pH' in component_name
        if is_pH:
            sim_unit_factor = 1.0
            obs_unit_factor = 1.0
        else:
            # Use consistent units (M) for comparison
            target_unit = 'M'
            sim_unit_factor = self._get_sim_unit_factor(target_unit)
            obs_unit_factor = self._get_obs_unit_factor(obs_component_name, target_unit)

        # Get simulation times as datetime
        sim_times = self.times
        sim_datetimes = [startdate + np.timedelta64(int(t), 'h') for t in sim_times]

        # Collect all observed vs simulated pairs
        all_observed = []
        all_simulated = []
        all_locations = []
        pairs_by_location: dict = {loc: {'obs': [], 'sim': []} for loc in self.config['obs_locs']}

        loc_dist = self.config['loc_dist']

        for loc in self.config['obs_locs']:
            # Get observations for this location
            df = chem_obs[chem_obs['Well'] == loc].copy()
            if obs_component_name not in df.columns:
                continue

            mask = df[obs_component_name].isna()
            df = df[~mask]

            if len(df) == 0:
                continue

            # Get the distance for this location
            loc_id = self.config['loc_name'].get(loc, '1')
            distance = None
            for d, lid in loc_dist.items():
                if lid == loc_id:
                    distance = d
                    break

            if distance is None or distance not in results:
                continue

            # Get simulation data for this distance
            sim_data = np.array(results[distance][component_name]) * sim_unit_factor

            # For each observation, find corresponding simulation value
            for _, row in df.iterrows():
                obs_date = row['Date']
                obs_value = row[obs_component_name] * obs_unit_factor

                # Find closest simulation time
                if hasattr(obs_date, 'to_numpy'):
                    obs_datetime = obs_date.to_numpy()
                else:
                    obs_datetime = np.datetime64(obs_date)

                # Calculate time differences
                time_diffs = [abs((obs_datetime - sd).astype('timedelta64[h]').astype(float))
                             for sd in sim_datetimes]
                closest_idx = np.argmin(time_diffs)

                # Only include if within max_time_diff_hours of a simulation output
                if time_diffs[closest_idx] <= max_time_diff_hours:
                    sim_value = sim_data[closest_idx]
                    all_observed.append(obs_value)
                    all_simulated.append(sim_value)
                    all_locations.append(loc)
                    pairs_by_location[loc]['obs'].append(obs_value)
                    pairs_by_location[loc]['sim'].append(sim_value)

        if len(all_observed) == 0:
            raise ValueError("No matching observation/simulation pairs found within "
                           f"{max_time_diff_hours} hours")

        all_observed = np.array(all_observed)
        all_simulated = np.array(all_simulated)

        # Calculate overall NSE
        obs_mean = np.mean(all_observed)
        ss_res = np.sum((all_observed - all_simulated) ** 2)  # Residual sum of squares
        ss_tot = np.sum((all_observed - obs_mean) ** 2)       # Total sum of squares

        if ss_tot == 0:
            nse = np.nan  # All observations are identical
        else:
            nse = 1 - (ss_res / ss_tot)

        # Calculate NSE by location
        nse_by_location = {}
        for loc, pairs in pairs_by_location.items():
            if len(pairs['obs']) > 1:
                obs_arr = np.array(pairs['obs'])
                sim_arr = np.array(pairs['sim'])
                loc_obs_mean = np.mean(obs_arr)
                loc_ss_res = np.sum((obs_arr - sim_arr) ** 2)
                loc_ss_tot = np.sum((obs_arr - loc_obs_mean) ** 2)
                if loc_ss_tot > 0:
                    nse_by_location[loc] = 1 - (loc_ss_res / loc_ss_tot)
                else:
                    nse_by_location[loc] = np.nan

        # Calculate additional statistics
        rmse = np.sqrt(np.mean((all_simulated - all_observed) ** 2))
        bias = np.mean(all_simulated - all_observed)

        return {
            'nse': float(nse),
            'nse_by_location': nse_by_location,
            'n_pairs': len(all_observed),
            'observed': all_observed,
            'simulated': all_simulated,
            'rmse': float(rmse),
            'bias': float(bias),
            'component': component_name,
            'obs_component': obs_component_name
        }

    # Mapping from water level observation column names to location IDs
    WATER_OBS_LOC_MAP = {
        'mzt11': '1', 'mzt13': '3', 'mzt15': '5',
        'mzt21': '1', 'mzt23': '3', 'mzt25': '5',
        'mcp11': '1', 'mcp13': '3', 'mcp15': '5',
    }

    # Base elevations for converting pressure to mASL
    BASE_ELEVATIONS = {
        'MZ': 2717.5,
        'MC': 2717.5,  # Adjust if different for MC
    }

    def _calculate_kge_for_water_levels(self, startdate: Any, water_obs: Any,
                                        results: dict, distances: List,
                                        max_time_diff_hours: float) -> Optional[dict]:
        """
        Internal method to calculate KGE for water levels.

        Converts simulated pressure to water elevation (mASL) and compares
        with observed groundwater elevations.

        Returns None if no valid observation-simulation pairs are found.
        """
        # Check for pressure component in simulation
        pressure_component = 'Liquid_Pressure [Pa]'
        first_distance = distances[0]
        if first_distance not in results or pressure_component not in results[first_distance]:
            return None

        # Get base elevation for this meander
        base_elev = self.BASE_ELEVATIONS.get(self.meander, 2717.5)

        # Determine date column name (handle both formats)
        date_col = 'Date and time' if 'Date and time' in water_obs.columns else 'Date'

        # Get simulation times as datetime
        sim_times = self.times
        sim_datetimes = [startdate + np.timedelta64(int(t), 'h') for t in sim_times]

        # Collect all observed vs simulated pairs
        all_observed = []
        all_simulated = []
        pairs_by_location: dict = {}

        loc_dist = self.config['loc_dist']

        # Find water level columns that match our locations
        for obs_col in water_obs.columns:
            if obs_col == date_col:
                continue

            # Get location ID from observation column name
            loc_id = self.WATER_OBS_LOC_MAP.get(obs_col.lower())
            if loc_id is None:
                continue

            # Find the distance for this location
            distance = None
            for d, lid in loc_dist.items():
                if lid == loc_id:
                    distance = d
                    break

            if distance is None or distance not in results:
                continue

            if pressure_component not in results[distance]:
                continue

            # Initialize location tracking
            if obs_col not in pairs_by_location:
                pairs_by_location[obs_col] = {'obs': [], 'sim': []}

            # Get simulated pressure and convert to mASL
            sim_pressure = np.array(results[distance][pressure_component])
            sim_elev = (sim_pressure - 101325.0) / (9.81 * 998.0) + base_elev

            # Match observations to simulations
            for _, row in water_obs.iterrows():
                obs_date = row[date_col]
                obs_value = row[obs_col]

                # Skip NaN observations
                if pd.isna(obs_value):
                    continue

                if hasattr(obs_date, 'to_numpy'):
                    obs_datetime = obs_date.to_numpy()
                else:
                    obs_datetime = np.datetime64(obs_date)

                time_diffs = [abs((obs_datetime - sd).astype('timedelta64[h]').astype(float))
                             for sd in sim_datetimes]
                closest_idx = np.argmin(time_diffs)

                if time_diffs[closest_idx] <= max_time_diff_hours:
                    sim_value = sim_elev[closest_idx]
                    # Skip non-finite values (inf, nan)
                    if np.isfinite(sim_value) and np.isfinite(obs_value):
                        all_observed.append(obs_value)
                        all_simulated.append(sim_value)
                        pairs_by_location[obs_col]['obs'].append(obs_value)
                        pairs_by_location[obs_col]['sim'].append(sim_value)

        if len(all_observed) == 0:
            return None

        all_observed = np.array(all_observed)
        all_simulated = np.array(all_simulated)

        # Calculate KGE components
        obs_mean = np.mean(all_observed)
        sim_mean = np.mean(all_simulated)
        obs_std = np.std(all_observed)
        sim_std = np.std(all_simulated)

        # Pearson correlation
        if obs_std > 0 and sim_std > 0:
            r = np.corrcoef(all_observed, all_simulated)[0, 1]
        else:
            r = np.nan

        # Variability ratio (alpha)
        alpha = sim_std / obs_std if obs_std > 0 else np.nan

        # Bias ratio (beta)
        beta = sim_mean / obs_mean if obs_mean != 0 else np.nan

        # Calculate KGE
        if not np.isnan(r) and not np.isnan(alpha) and not np.isnan(beta):
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge = np.nan

        # Calculate NSE
        ss_res = np.sum((all_observed - all_simulated) ** 2)
        ss_tot = np.sum((all_observed - obs_mean) ** 2)
        nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Calculate RMSE
        rmse = np.sqrt(np.mean((all_simulated - all_observed) ** 2))

        # Calculate KGE by location
        kge_by_location = {}
        for loc, pairs in pairs_by_location.items():
            if len(pairs['obs']) > 1:
                obs_arr = np.array(pairs['obs'])
                sim_arr = np.array(pairs['sim'])
                loc_obs_mean = np.mean(obs_arr)
                loc_sim_mean = np.mean(sim_arr)
                loc_obs_std = np.std(obs_arr)
                loc_sim_std = np.std(sim_arr)

                if loc_obs_std > 0 and loc_sim_std > 0:
                    loc_r = np.corrcoef(obs_arr, sim_arr)[0, 1]
                    loc_alpha = loc_sim_std / loc_obs_std
                    loc_beta = loc_sim_mean / loc_obs_mean if loc_obs_mean != 0 else np.nan
                    if not np.isnan(loc_r) and not np.isnan(loc_beta):
                        kge_by_location[loc] = 1 - np.sqrt(
                            (loc_r - 1)**2 + (loc_alpha - 1)**2 + (loc_beta - 1)**2
                        )

        return {
            'kge': float(kge) if not np.isnan(kge) else np.nan,
            'nse': float(nse) if not np.isnan(nse) else np.nan,
            'rmse': float(rmse),
            'kge_components': {
                'r': float(r) if not np.isnan(r) else np.nan,
                'alpha': float(alpha) if not np.isnan(alpha) else np.nan,
                'beta': float(beta) if not np.isnan(beta) else np.nan
            },
            'kge_by_location': kge_by_location,
            'n_pairs': len(all_observed),
            'observed': all_observed,
            'simulated': all_simulated,
            'component': 'Water_Level [mASL]',
            'obs_component': 'GW Elevation'
        }

    def _calculate_kge_for_component(self, component_name: str, startdate: Any,
                                      chem_obs: Any, results: dict,
                                      distances: List, max_time_diff_hours: float) -> Optional[dict]:
        """
        Internal method to calculate KGE for a single component.

        Returns None if no valid observation-simulation pairs are found.
        """
        # Get observation component name
        obs_component_name = self.COMPONENT_TO_OBS_MAP.get(component_name)
        if obs_component_name is None:
            return None

        # Check if observation component exists in data
        if obs_component_name not in chem_obs.columns:
            return None

        # Check if component exists in simulation results
        first_distance = distances[0]
        if first_distance not in results or component_name not in results[first_distance]:
            return None

        # Determine unit conversion factors
        is_pH = 'pH' in component_name
        if is_pH:
            sim_unit_factor = 1.0
            obs_unit_factor = 1.0
        else:
            target_unit = 'M'
            sim_unit_factor = self._get_sim_unit_factor(target_unit)
            obs_unit_factor = self._get_obs_unit_factor(obs_component_name, target_unit)

        # Get simulation times as datetime
        sim_times = self.times
        sim_datetimes = [startdate + np.timedelta64(int(t), 'h') for t in sim_times]

        # Collect all observed vs simulated pairs
        all_observed = []
        all_simulated = []
        pairs_by_location: dict = {loc: {'obs': [], 'sim': []} for loc in self.config['obs_locs']}

        loc_dist = self.config['loc_dist']

        for loc in self.config['obs_locs']:
            df = chem_obs[chem_obs['Well'] == loc].copy()
            if obs_component_name not in df.columns:
                continue

            mask = df[obs_component_name].isna()
            df = df[~mask]

            if len(df) == 0:
                continue

            loc_id = self.config['loc_name'].get(loc, '1')
            distance = None
            for d, lid in loc_dist.items():
                if lid == loc_id:
                    distance = d
                    break

            if distance is None or distance not in results:
                continue

            if component_name not in results[distance]:
                continue

            sim_data = np.array(results[distance][component_name]) * sim_unit_factor

            for _, row in df.iterrows():
                obs_date = row['Date']
                obs_value = row[obs_component_name] * obs_unit_factor

                if hasattr(obs_date, 'to_numpy'):
                    obs_datetime = obs_date.to_numpy()
                else:
                    obs_datetime = np.datetime64(obs_date)

                time_diffs = [abs((obs_datetime - sd).astype('timedelta64[h]').astype(float))
                             for sd in sim_datetimes]
                closest_idx = np.argmin(time_diffs)

                if time_diffs[closest_idx] <= max_time_diff_hours:
                    sim_value = sim_data[closest_idx]
                    # Skip non-finite values (inf, nan)
                    if np.isfinite(sim_value) and np.isfinite(obs_value):
                        all_observed.append(obs_value)
                        all_simulated.append(sim_value)
                        pairs_by_location[loc]['obs'].append(obs_value)
                        pairs_by_location[loc]['sim'].append(sim_value)

        if len(all_observed) == 0:
            return None

        all_observed = np.array(all_observed)
        all_simulated = np.array(all_simulated)

        # Calculate KGE components
        obs_mean = np.mean(all_observed)
        sim_mean = np.mean(all_simulated)
        obs_std = np.std(all_observed)
        sim_std = np.std(all_simulated)

        # Pearson correlation
        if obs_std > 0 and sim_std > 0:
            r = np.corrcoef(all_observed, all_simulated)[0, 1]
        else:
            r = np.nan

        # Variability ratio (alpha)
        alpha = sim_std / obs_std if obs_std > 0 else np.nan

        # Bias ratio (beta)
        beta = sim_mean / obs_mean if obs_mean != 0 else np.nan

        # Calculate KGE
        if not np.isnan(r) and not np.isnan(alpha) and not np.isnan(beta):
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge = np.nan

        # Calculate NSE
        ss_res = np.sum((all_observed - all_simulated) ** 2)
        ss_tot = np.sum((all_observed - obs_mean) ** 2)
        nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Calculate RMSE
        rmse = np.sqrt(np.mean((all_simulated - all_observed) ** 2))

        # Calculate KGE by location
        kge_by_location = {}
        for loc, pairs in pairs_by_location.items():
            if len(pairs['obs']) > 1:
                obs_arr = np.array(pairs['obs'])
                sim_arr = np.array(pairs['sim'])
                loc_obs_mean = np.mean(obs_arr)
                loc_sim_mean = np.mean(sim_arr)
                loc_obs_std = np.std(obs_arr)
                loc_sim_std = np.std(sim_arr)

                if loc_obs_std > 0 and loc_sim_std > 0:
                    loc_r = np.corrcoef(obs_arr, sim_arr)[0, 1]
                    loc_alpha = loc_sim_std / loc_obs_std
                    loc_beta = loc_sim_mean / loc_obs_mean if loc_obs_mean != 0 else np.nan
                    if not np.isnan(loc_r) and not np.isnan(loc_beta):
                        kge_by_location[loc] = 1 - np.sqrt(
                            (loc_r - 1)**2 + (loc_alpha - 1)**2 + (loc_beta - 1)**2
                        )

        return {
            'kge': float(kge) if not np.isnan(kge) else np.nan,
            'nse': float(nse) if not np.isnan(nse) else np.nan,
            'rmse': float(rmse),
            'kge_components': {
                'r': float(r) if not np.isnan(r) else np.nan,
                'alpha': float(alpha) if not np.isnan(alpha) else np.nan,
                'beta': float(beta) if not np.isnan(beta) else np.nan
            },
            'kge_by_location': kge_by_location,
            'n_pairs': len(all_observed),
            'observed': all_observed,
            'simulated': all_simulated,
            'component': component_name,
            'obs_component': obs_component_name
        }

    def calculate_kge(self, startdate: Any, chem_obs: Any = None,
                      component_name: Optional[str] = None,
                      water_obs: Any = None,
                      results: Optional[dict] = None,
                      distances: Optional[List] = None,
                      depths: Optional[List[float]] = None,
                      max_time_diff_hours: float = 12.0,
                      print_summary: bool = True) -> dict:
        """
        Calculate Kling-Gupta Efficiency (KGE) for simulation output parameters.

        If component_name is None, calculates KGE for ALL simulation components
        that have matching observational data. Otherwise, calculates for the
        specified component only. If water_obs is provided, also calculates
        KGE for water levels.

        KGE = 1 - sqrt((r - 1)² + (α - 1)² + (β - 1)²)

        Where:
        - r = Pearson correlation coefficient
        - α = variability ratio (σ_sim / σ_obs)
        - β = bias ratio (μ_sim / μ_obs)

        KGE interpretation:
        - KGE = 1: Perfect match
        - KGE > -0.41: Model is better than using mean of observations
        - KGE < -0.41: Mean of observations is a better predictor

        Args:
            startdate: Starting date for simulation (e.g., np.datetime64('2019-05-01'))
            chem_obs: Chemical observations DataFrame with 'Date' and 'Well' columns.
                     Can be None if only calculating water levels.
            component_name: Optional specific component name (e.g., 'Total_Fe++ [M]').
                           If None, calculates for all components with observations.
            water_obs: Optional water level observations DataFrame with 'Date and time'
                      column and well columns (e.g., 'mzt11', 'mzt13', 'mzt15').
                      Values should be in mASL.
            results: Optional dict of results by distance. If None, calls get_histories().
            distances: Optional distance points. If None, uses self.config['distances'].
            depths: Optional depths for get_histories() if extracting data.
            max_time_diff_hours: Maximum time difference (hours) for matching
                                 observations to simulations. Default is 12 hours.
            print_summary: Whether to print a summary table of results. Default True.

        Returns:
            Dictionary containing:
            - 'components': Dict mapping component names to their KGE results
            - 'summary': DataFrame with KGE, NSE, RMSE, n for each component
            - 'n_components': Number of components with valid KGE calculations

        Example:
            # Calculate KGE for all components with observations
            stats = processor.calculate_kge(
                startdate=np.datetime64('2019-05-01'),
                chem_obs=obs_df
            )

            # Include water levels
            stats = processor.calculate_kge(
                startdate=np.datetime64('2019-05-01'),
                chem_obs=obs_df,
                water_obs=water_obs_df
            )

            # Water levels only
            stats = processor.calculate_kge(
                startdate=np.datetime64('2019-05-01'),
                water_obs=water_obs_df
            )
        """
        # Validate inputs
        if chem_obs is None and water_obs is None:
            raise ValueError("At least one of chem_obs or water_obs must be provided")

        # Use defaults from processor if not provided
        if distances is None:
            distances = list(self.config['distances'])

        # Determine which components to process
        if component_name is not None:
            # Single component mode
            components_to_check = [component_name]
        elif chem_obs is not None:
            # All components mode - get from COMPONENT_TO_OBS_MAP
            components_to_check = list(self.COMPONENT_TO_OBS_MAP.keys())
        else:
            components_to_check = []

        # Build list of components to extract (including pressure for water levels)
        components_to_extract = []
        if chem_obs is not None:
            available_sim_components = list(self.component_list) if hasattr(self, 'component_list') else []
            components_to_extract = [c for c in components_to_check if c in available_sim_components]

        # Add pressure component if water levels requested
        if water_obs is not None:
            if 'Liquid_Pressure [Pa]' not in components_to_extract:
                components_to_extract.append('Liquid_Pressure [Pa]')

        # Extract results if not provided
        if results is None:
            if not components_to_extract:
                raise ValueError("No matching components found in simulation output")
            results, _ = self.get_histories(depths=depths, components=components_to_extract)

        # Calculate KGE for each chemical component
        component_results = {}
        summary_data = []

        if chem_obs is not None:
            for comp in components_to_check:
                result = self._calculate_kge_for_component(
                    comp, startdate, chem_obs, results, distances, max_time_diff_hours
                )
                if result is not None:
                    component_results[comp] = result
                    summary_data.append({
                        'Component': comp,
                        'Obs': result['obs_component'],
                        'KGE': result['kge'],
                        'NSE': result['nse'],
                        'RMSE': result['rmse'],
                        'r': result['kge_components']['r'],
                        'α': result['kge_components']['alpha'],
                        'β': result['kge_components']['beta'],
                        'n': result['n_pairs']
                    })

        # Calculate KGE for water levels if provided
        if water_obs is not None:
            water_result = self._calculate_kge_for_water_levels(
                startdate, water_obs, results, distances, max_time_diff_hours
            )
            if water_result is not None:
                component_results['Water_Level [mASL]'] = water_result
                summary_data.append({
                    'Component': 'Water_Level [mASL]',
                    'Obs': water_result['obs_component'],
                    'KGE': water_result['kge'],
                    'NSE': water_result['nse'],
                    'RMSE': water_result['rmse'],
                    'r': water_result['kge_components']['r'],
                    'α': water_result['kge_components']['alpha'],
                    'β': water_result['kge_components']['beta'],
                    'n': water_result['n_pairs']
                })

        if len(component_results) == 0:
            raise ValueError("No components with matching observation data found")

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Print summary if requested
        if print_summary:
            print(f"\n{'='*80}")
            print(f"KGE Summary - {len(component_results)} components with observations")
            print(f"{'='*80}")
            print(f"{'Component':<25} {'Obs':<12} {'KGE':>8} {'NSE':>8} {'RMSE':>10} {'r':>6} {'α':>6} {'β':>6} {'n':>5}")
            print(f"{'-'*80}")
            for _, row in summary_df.iterrows():
                print(f"{row['Component']:<25} {row['Obs']:<12} {row['KGE']:>8.3f} {row['NSE']:>8.3f} "
                      f"{row['RMSE']:>10.2e} {row['r']:>6.3f} {row['α']:>6.3f} {row['β']:>6.3f} {row['n']:>5}")
            print(f"{'='*80}\n")

        # For single component mode, also return the direct results for convenience
        if component_name is not None and component_name in component_results:
            return {
                **component_results[component_name],
                'components': component_results,
                'summary': summary_df,
                'n_components': len(component_results)
            }

        return {
            'components': component_results,
            'summary': summary_df,
            'n_components': len(component_results)
        }

    # =========================================================================
    # Thermodynamic Calculations
    # =========================================================================

    @staticmethod
    def calc_fh_dG(data: dict, time_t: float) -> np.ndarray:
        """
        Calculate Gibbs free energy for ferrihydrite reduction.

        Rxn: 1.00 Ac- + 8.00 FHY + 15.00 H+ = 8.00 Fe++ + 2.00 HCO3- + 20.00 H2O

        Args:
            data: Dictionary with species concentrations
            time_t: Time key

        Returns:
            dGr values
        """
        dG0 = -612.0  # kJ per mol Ac-

        fe2 = np.power(data['Free_Fe++ [M]'][time_t], 8)
        hco3 = np.power(data['Free_HCO3- [M]'][time_t], 2)
        proton = np.power(np.power(10, -1 * data['pH'][time_t]), 15)
        ac = np.power(data['Free_Ac- [M]'][time_t], 1)

        q = np.divide(np.multiply(fe2, hco3), np.multiply(proton, ac))

        R = 8.314e-3  # kJ / (K * mol)
        return dG0 + R * 273.15 * np.log(q)

    @staticmethod
    def calc_gt_dG(data: dict, time_t: float) -> np.ndarray:
        """
        Calculate Gibbs free energy for goethite reduction.

        Rxn: 1.00 Ac- + 8.00 GT + 15.00 H+ = 8.00 Fe++ + 2.00 HCO3- + 12.00 H2O

        Args:
            data: Dictionary with species concentrations
            time_t: Time key

        Returns:
            dGr values
        """
        dG0 = -464  # kJ per mol Ac-

        fe2 = np.power(data['Free_Fe++ [M]'][time_t], 8)
        hco3 = np.power(data['Free_HCO3- [M]'][time_t], 2)
        proton = np.power(np.power(10, -1 * data['pH'][time_t]), 15)
        ac = np.power(data['Free_Ac- [M]'][time_t], 1)

        q = np.divide(np.multiply(fe2, hco3), np.multiply(proton, ac))

        R = 8.314e-3
        return dG0 + R * 273.15 * np.log(q)

    @staticmethod
    def calc_sulf_dG(data: dict, time_t: float) -> np.ndarray:
        """
        Calculate Gibbs free energy for sulfate reduction.

        Rxn: 1.00 Ac- + 1.00 SO4-- = 1.00 HS- + 2.00 HCO3-

        Args:
            data: Dictionary with species concentrations and activity coefficients
            time_t: Time key

        Returns:
            dGr values
        """
        dG0 = -48.1  # kJ per mol Ac-, Kocar & Fendorf 2009

        sulfate = np.power(data['Free_SO4-- [M]'][time_t] * data['Gamma_SO4--'][time_t], 1)
        hco3 = np.power(data['Free_HCO3- [M]'][time_t] * data['Gamma_HCO3-'][time_t], 2)
        ac = np.power(data['Free_Ac- [M]'][time_t] * data['Gamma_Ac-'][time_t], 1)
        hs = np.power(data['Free_HS- [M]'][time_t] * data['Gamma_HS-'][time_t], 1)

        q = np.divide(np.multiply(hs, hco3), np.multiply(sulfate, ac))

        R = 8.314e-3
        return dG0 + R * 298.15 * np.log(q)

    @staticmethod
    def calc_FT(dgr: np.ndarray, m: float, chi: float) -> np.ndarray:
        """
        Calculate thermodynamic factor.

        Args:
            dgr: Gibbs free energy of reaction
            m: ATP yield parameter
            chi: Average stoichiometric number

        Returns:
            Thermodynamic factor
        """
        return 1 - np.exp((dgr + m * 50) / (chi * 8.314e-3 * 273.15))

    # =========================================================================
    # Grid Building Utilities
    # =========================================================================

    @staticmethod
    def get_distance(list1: np.ndarray, list2: np.ndarray) -> List:
        """
        Calculate distances between points in two lists.

        Args:
            list1: Array of [row, col] points
            list2: Array of [row, col] points

        Returns:
            List of [(point1, point2), distance] pairs
        """
        distance = []
        for l1 in range(len(list1)):
            for l2 in range(len(list2)):
                d = math.sqrt((list1[l1][0] - list2[l2][0])**2 +
                             (list1[l1][1] - list2[l2][1])**2)
                distance.append([([list1[l1][0], list1[l1][1]],
                                 [list2[l2][0], list2[l2][1]]), d])
        return distance

    @staticmethod
    def make_pflo_mat(vec: np.ndarray, dim: Tuple[int, int]) -> np.ndarray:
        """
        Convert 1D vector to 2D matrix for PFLOTRAN.

        Elements are added bottom-to-top, left-to-right to match
        PFLOTRAN's material ID ordering.

        Args:
            vec: 1D array
            dim: Target dimensions (rows, cols)

        Returns:
            2D array
        """
        i = 0
        rows, cols = dim
        mat = np.zeros(dim)
        for rr in range(rows, 0, -1):
            for cc in range(cols):
                mat[rr-1, cc] = vec[i]
                i += 1
        return mat.astype('float')

    @staticmethod
    def make_3d_mat(vec: np.ndarray, dim: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert 1D vector to 3D matrix for PFLOTRAN.

        Args:
            vec: 1D array
            dim: Target dimensions (nx, ny, nz)

        Returns:
            3D array
        """
        cc = 0
        rows, cols, nz = dim
        mat = np.zeros(dim)
        for jj in range(cols):
            for kk in range(nz, 0, -1):
                for ii in range(rows, 0, -1):
                    mat[ii-1, jj, kk-1] = vec[cc]
                    cc += 1
        return mat.astype('float')

    @staticmethod
    def find_cells(mat1d: np.ndarray, eq: str, val: float) -> np.ndarray:
        """
        Find cell indices matching a value.

        Args:
            mat1d: 1D array of material IDs
            eq: "equal" or "not" for matching condition
            val: Value to match

        Returns:
            Array of matching indices
        """
        if eq == "equal":
            cells = [i for i in range(len(mat1d)) if mat1d[i] == val]
        else:
            cells = [i for i in range(len(mat1d)) if mat1d[i] != val]
        return np.asarray(cells)

    @staticmethod
    def assign_inactive(elev: float, domain: np.ndarray) -> np.ndarray:
        """
        Assign inactive cells based on elevation threshold.

        Args:
            elev: Elevation threshold (cells below are inactive)
            domain: 2D DEM array

        Returns:
            Material ID array (0=inactive, 1=active)
        """
        ny, nx = domain.shape
        mat_id = np.ones((ny, nx), dtype=int)

        for xx in range(nx):
            for yy in range(ny):
                if domain[yy, xx] < elev:
                    mat_id[yy, xx] = 0

        return mat_id

    @staticmethod
    def deactivate_range(points: List, direction: str, mat_id: np.ndarray) -> np.ndarray:
        """
        Deactivate cells above or below a line.

        Args:
            points: Two points defining line [[x1,y1], [x2,y2]]
            direction: 'above' or 'below'
            mat_id: Material ID matrix

        Returns:
            Updated material ID matrix
        """
        x1, y1 = points[0]
        x2, y2 = points[1]
        yl = mat_id.shape[0]

        m = -(y2 - y1) / (x2 - x1)
        m = np.ceil(1 / m)
        sign = np.abs(m) / m
        dstep = 1
        dy = 0

        if direction == 'above':
            for xx in range(x1, x2):
                for yy in range(0, int(y1 + dy)):
                    mat_id[yy, xx] = 0
                if dstep % m == 0:
                    dy = dy - sign
                dstep += 1
        elif direction == 'below':
            for xx in range(x1, x2):
                for yy in range(int(y1 + dy), yl):
                    mat_id[yy, xx] = 0
                if dstep % m == 0:
                    dy = dy - sign
                dstep += 1

        return mat_id

    @staticmethod
    def assign_BC_cells(mat_id: np.ndarray, idn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign boundary condition cells around active region.

        Args:
            mat_id: Material ID matrix
            idn: ID number for BC cells

        Returns:
            Tuple of (updated mat_id, faces, coords)
        """
        ny, nx = mat_id.shape
        faces = np.zeros_like(mat_id)
        coords = []

        for xx in range(nx):
            for yy in range(ny):
                if mat_id[yy, xx] == 1 and mat_id[yy-1, xx] == 0:
                    if yy == 0:
                        mat_id[yy, xx] = idn + 2
                        faces[yy, xx] = 3
                        coords.append([yy, xx, 1])
                    else:
                        mat_id[yy-1, xx] = idn
                        faces[yy-1, xx] = 4
                        coords.append([yy-1, xx, 1])

                if mat_id[yy, xx] == 0 and mat_id[yy-1, xx] == 1:
                    mat_id[yy, xx] = idn
                    faces[yy, xx] = 3
                    coords.append([yy, xx, 1])

                if mat_id[yy, xx] == 1 and mat_id[yy, xx-1] == 0:
                    if xx == 0:
                        mat_id[yy, xx] = idn + 2
                        faces[yy, xx] = 2
                        coords.append([yy, xx, 1])
                    else:
                        mat_id[yy, xx-1] = idn
                        faces[yy, xx-1] = 1
                        coords.append([yy, xx-1, 1])

                if mat_id[yy, xx] == 0 and mat_id[yy, xx-1] == 1:
                    mat_id[yy, xx] = idn
                    faces[yy, xx] = 2
                    coords.append([yy, xx, 1])

        return mat_id, faces, np.asarray(coords)


class Region:
    """
    HDF5 Region writer for PFLOTRAN grid files.

    Creates region definitions in HDF5 format from text files
    containing cell ID and face ID pairs.
    """

    def __init__(self, region_group: h5py.Group, region_name: str, filename: str):
        """
        Initialize Region writer.

        Args:
            region_group: HDF5 group for regions
            region_name: Name of the region
            filename: Path to text file with cell/face data
        """
        self.region_name = region_name
        self.filename = filename
        self.group = region_group.create_group(region_name)

    def write_region(self, n: int):
        """
        Write region data to HDF5.

        Args:
            n: Number of cells in region
        """
        cell_id_array = np.zeros(n, dtype='=i4')
        face_id_array = np.zeros(n, dtype='=i4')

        with open(self.filename) as f:
            count = 0
            while True:
                s = f.readline()
                if len(s) < 2:
                    break
                w = s.split()
                cell_id_array[count] = int(float(w[0]))
                face_id_array[count] = int(float(w[1]))
                count += 1

        iarray = np.zeros(count, dtype='=i4')
        iarray[:count] = cell_id_array[:count]
        self.group.create_dataset('Cell Ids', data=iarray)

        iarray[:count] = face_id_array[:count]
        self.group.create_dataset('Face Ids', data=iarray)

        print(f'done with Region: {self.region_name}')

    # Alias for backwards compatibility
    writeRegion = write_region


# Utility functions for legend creation
def make_legend(fig: Any, labels: List[str], xpos: float = 1.0) -> None:
    """
    Create a custom legend with viridis colors.

    Args:
        fig: Matplotlib figure
        labels: List of legend labels
        xpos: X position for legend
    """
    from matplotlib.lines import Line2D

    cmap = mpl.colormaps['viridis']
    cmaplist = [cmap(i) for i in np.arange(0, 1, 0.2)]

    custom_lines = [Line2D([0], [0], color=c, lw=4) for c in cmaplist[:len(labels)]]

    fig.legend(custom_lines, labels, bbox_to_anchor=(xpos, 0.5),
               loc='center', frameon=False, ncol=1)


def label_panel(ax: Axes, xpos: float, ypos: float, label: str, fontsize: int = 12) -> None:
    """
    Add a panel label to axes.

    Args:
        ax: Matplotlib axes
        xpos: X position (0-1)
        ypos: Y position (0-1)
        label: Label text
        fontsize: Font size
    """
    ax.text(xpos, ypos, label,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontweight='bold',
            fontsize=fontsize)
