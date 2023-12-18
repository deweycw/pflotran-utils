# %%
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
#import encapsulation.factory.parameters as Params
#import h5_output.factory.h5_output_class as H5Class
from h5_output.calc.cross_section import CrossSection 

# %%
data_dir = '/Users/christiandewey/Code/meander-models/pflotran-files/2D/mzt2/spin/'
file_location = data_dir + 'pflotran-spin-2.h5'

xsection = CrossSection(file_location,'x')
xsection.print_components(include='Rate',exclude='Sorbed')
xsection.get_cells()
#xsection.get_material_ids(show_inactive=True)
component = "Total_HCO3- [M]"

xsection.plot_at_time(component=component,time_t=1,show_unsat=False)
xsection.plot_at_time(component=component,time_t=10,show_unsat=False)
xsection.plot_at_time(component=component,time_t=15,show_unsat=False)
xsection.plot_at_time(component=component,time_t=20,show_unsat=False)
xsection.plot_at_time(component=component,time_t=25,show_unsat=False)
xsection.plot_at_time(component=component,time_t=50,show_unsat=True)
#xsection.plot_history_at_cell(component='Total_Tracer [M]',cell_loc=(50,18))
#xsection.plot_history_at_m_coords(component=component,meter_coords=(50,1.8))
# %%
