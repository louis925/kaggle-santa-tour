import sys
from util_io import init
from util_cost import cal_total, cal_total_preference, cal_total_accounting

# params #
path_init_conf = '../output/m24-improved-random.csv'
#path_init_conf = '../output/m24-improved-2.csv'

if len(sys.argv) == 2:
    path_init_conf = sys.argv[1]

print('Score file:', path_init_conf)
assigned_day, family_on_day, occupancy = init(path_conf=path_init_conf)
etotal = cal_total(assigned_day, occupancy)
print('Init config cost: {}'.format(etotal))
print('Preference cost:', cal_total_preference(assigned_day))
print('Accounting cost:', cal_total_accounting(occupancy))
