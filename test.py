# %%
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
#import encapsulation.factory.parameters as Params
#import h5_output.factory.h5_output_class as H5Class
from h5_output.calc.cross_section import CrossSection 
import datetime
from numpy import savetxt
import numpy as np
import matplotlib.pyplot as plt
def get_hours(target_date: datetime.datetime, year:str):
    if year == '2019':
        start_date = datetime.datetime(2019, 4, 21) #, 00, 00)
        end_date = datetime.datetime(2019, 10, 2) #, 00, 00)
    elif year == '2018':
        start_date = datetime.datetime(2018, 4, 1) #, 00, 00)
        end_date = datetime.datetime(2018, 10, 31) #, 00, 00)

    diff = target_date - start_date

    days_diff = diff.days
    hours_diff = days_diff * 24 

    return hours_diff



data_dir = '/Users/christiandewey/Code/meander-models/pflotran-files/2D/mcp-18/'
file_location = data_dir + 'pflotran-spin-2-n.h5'


if '18' in str.split(data_dir,'/')[-2]:
    year = 2018
elif '19' in str.split(data_dir,'/')[-2]:
    year = 2019

components = ["Total_DOC- [M]", "Total_Ca++ [M]", "Total_Fe++ [M]", "Total_HCO3- [M]", "pH",
               "JB_Fh_Acetate_Sandbox_Rate [mol-Ac_sec]", "JB_Gt_Acetate_Sandbox_Rate [mol-Ac_sec]",
               "JB_Nitrate_Acetate_Sandbox_Rate [mol-Ac_sec]", "JB_Sulfate_Acetate_Sandbox_Rate [mol-Ac_sec]"]


xsection = CrossSection(file_location,'x')
xsection.get_cells()
xsection.print_components(include='Rate')

if 'mzt' in str.split(data_dir,'/')[-2]:
    distances = [1.0, 16, 27, 40, 50] #m 
else:
    distances = [0.5, 16, 31, 46, 60.]

depths = [1.7,2.0,2.1,2.4,2.5]
locs = [(i,d) for i, d in zip(distances,depths)]
date_list = [datetime.datetime(year, 5,21),datetime.datetime(year, 6, 5) ]

times = [18000,19000,20000]

output_dir = '/Users/christiandewey/Library/CloudStorage/Dropbox/Manuscripts/DOC & Fe Cycling across East River Meanders/model-output/' + str.split(data_dir,'/')[-2] + '/'

xsection.get_material_ids(locs =locs)
for component in components:
    if 'sim' in file_location:
        times = [get_hours(date_time,str(year)) for date_time in date_list]
        for t in times:
            print(t)
            xsection.plot_at_time(component=component,time_t=t,locs=locs,show_unsat=False)
            df = xsection.get_snapshot_all_cells(component,t)
            
    else:
        for t in times:
            xsection.plot_at_time(component=component,time_t=t,show_unsat=False)



    '''i = 1
    for loc in locs:
        df = xsection.get_history_at_m_coords(component=component,meter_coords=loc)
        savetxt(output_dir + component + '-loc' + str(i) + '.csv', df, delimiter=",")
        i = i + 1 '''
# %%
