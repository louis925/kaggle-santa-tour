import sys

from util_io import init, dump_conf
from util_cost import cal_total
from util_check import check_valid_all
from util_events import correct_invalid_up

# constants #
N_families = 5000
N_days = 100
# constants #

# params #
lower_occupancy = 125
upper_occupancy = 300
# params #

assert len(sys.argv) == 3, 'Usage: %s path_input path_output' % sys.argv[0]
path_input = sys.argv[1]
path_output = sys.argv[2]

assigned_day, family_on_day, occupancy = init(path_conf=path_input)
if check_valid_all(occupancy, lower_occupancy=lower_occupancy, upper_occupancy=upper_occupancy):
    print('All days are valid. do nothing.')
else:
    print('Not all valid. Doing corrections using greedy algo')
    etotal0 = cal_total(assigned_day, occupancy)
    correct_invalid_up(assigned_day, family_on_day, occupancy, upper_occupancy=upper_occupancy)
    etotal1 = cal_total(assigned_day, occupancy)
    print('finished. Cost changed from %.5f to %.5f.' % (etotal0, etotal1))
    print('Saving output to %s.' % path_output)
    dump_conf(assigned_day, path_output)
