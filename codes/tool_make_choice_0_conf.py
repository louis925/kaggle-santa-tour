import sys
from util_io import dump_conf
from util_cost import choices, n_people, cal_total

# constants #
N_families = 5000
N_days = 100
# constants #

assigned_day = []
occupancy = [0 for i in range(N_days+1)]
for i in range(N_families):
    assigned_day.append(choices[i,0])
    occupancy[choices[i,0]] += n_people[i]

path_out = input('Enter the output path: ')
dump_conf(assigned_day, path_out)
