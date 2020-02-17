"""
select one family and then put them into diff (valid) day.
if cost reduce, accept it; otherwise, accept by probability of exp(-cost_diff/temperature)
"""

import numpy as np
import pandas as pd

from util_io import init, finalize, dump_conf
from util_cost import cal_total, cal_diff_1
from util_cost import n_people
from util_check import check_valid_all, check_valid_move
from util_events import move_family

# constants #
N_families = 5000
N_days = 100
# constants #

# params #
#path_init_conf = '../output/answer_temp_350k.csv'
path_init_conf = '../output/m01-improved.csv'
path_dump_improved = '../output/m03-improved.csv' # lowest cost
path_dump_continue = '../output/m03-continue.csv' # current state

temp_upper = 50
temp_lower = 5
temp_diff = 5
temp_cycles = 2
temp_list = (
    (
        list(range(temp_upper, temp_lower, -temp_diff)) +
        list(range(temp_lower, temp_upper, temp_diff))
    ) * temp_cycles
)

N_iters = 1000000
N_dumps = 100000 # dump every iters
# params #

# init
assigned_day, family_on_day, occupancy = init(path_conf=path_init_conf)
etotal = cal_total(assigned_day, occupancy)
etotal_low = etotal
print('Init config cost:', etotal)

print('N of temp:', len(temp_list), temp_list)

for temp in temp_list:
    # init vars
    etotal = cal_total(assigned_day, occupancy)
    is_change = False

    print('At temp = %.3f. init cost: %.5f' % (temp, etotal))

    for iters in range(N_iters):
        # dump conf
        if is_change and iters % N_dumps == 0:
            dump_conf(assigned_day, path_dump_continue)
            is_change = False

        # pick family
        ifamily = np.random.randint(N_families)
        nfamily = n_people[ifamily]

        # pick day
        day0 = assigned_day[ifamily]
        day1 = np.random.randint(1, N_days+1)

        # check validity
        if day0 == day1:
            continue
        if not check_valid_move(occupancy, day0, day1, nfamily):
            continue

        # determine acceptance
        ediff = cal_diff_1(ifamily, occupancy, day0, day1)
        if ediff > 0 and np.random.random() >= np.exp(-ediff/temp): # TODO: see if > or >=
            continue
        else:
            move_family(ifamily, nfamily, day0, day1, assigned_day, family_on_day, occupancy)
            is_change = True

        # update etotal and check improvement
        etotal = etotal + ediff
        if etotal < etotal_low:
            print('Got improvement from {} to {}.'.format(etotal_low, etotal))
            dump_conf(assigned_day, path_dump_improved)
            etotal_low = etotal

# finalize
print('Final score:', cal_total(assigned_day, occupancy))
finalize(assigned_day, path_dump=path_dump_continue)
