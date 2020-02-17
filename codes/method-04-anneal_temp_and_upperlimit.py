"""
select one family and then put them into diff (valid) day.
if cost reduce, accept it; otherwise, accept by probability of exp(-cost_diff/temperature)
anneal in temp. and upper_occupancy.
NOTICE: upper_occupancy constraint may not valid at some point.
NOTICE: if found larger than upper_occupancy, move family no matter what.
"""

import numpy as np
import pandas as pd

from util_io import init, finalize, dump_conf
from util_cost import cal_total, cal_diff_1
from util_cost import n_people
from util_check import check_valid_all, check_valid_move, deep_check
from util_events import move_family, correct_invalid_up

# constants #
N_families = 5000
N_days = 100
# constants #

# params #
path_init_conf = '../output/answer_temp_350k.csv'
path_dump_improved = '../output/m04-improved.csv' # lowest cost
path_dump_continue = '../output/m04-continue.csv' # current state

N_runs = 51
list_temp = [t for t in range(55, 4, -1)]
list_upper_occupancy = [u for u in range(350, 299, -1)]
list_N_iters = [1000000 for i in range(30)] + [2000000 + 1000000*i for i in range(21)]

assert len(list_temp) == N_runs, 'size list_temp != N_runs: %d, %d' % (len(list_temp), N_runs)
assert len(list_upper_occupancy) == N_runs, 'size list_upper_occupancy != N_runs: %d, %d' % (len(list_upper_occupancy), N_runs)
assert len(list_N_iters) == N_runs, 'size list_N_iters != N_runs: %d, %d' % (len(list_N_iters), N_runs)

N_dumps = 5000000 # dump every iters
# params #

# init
print('list_temp:')
print(list_temp)
print('list_upper_occupancy:')
print(list_upper_occupancy)
print('list_N_iters:')
print(list_N_iters)

assigned_day, family_on_day, occupancy = init(path_conf=path_init_conf)
etotal_low = cal_total(assigned_day, occupancy)
print('Init config cost:', etotal_low)

for i_run in range(N_runs):
    # params set up
    temp = list_temp[i_run]
    upper_occupancy = list_upper_occupancy[i_run]
    N_iters = list_N_iters[i_run]

    # correct invalid days (if > upper_occ)
    correct_invalid_up(assigned_day, family_on_day, occupancy, upper_occupancy=upper_occupancy)
    deep_check(assigned_day, family_on_day, occupancy, upper_occupancy=upper_occupancy)

    # init vars
    etotal = cal_total(assigned_day, occupancy)
    is_change = False
   
    print('At temp = %.1f, upper_occ = %d, N_iters = %d. init cost: %.5f' % (temp, upper_occupancy, N_iters, etotal))

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
        if not check_valid_move(occupancy, day0, day1, nfamily, upper_occupancy=upper_occupancy):
            continue

        # determine acceptance
        ediff = cal_diff_1(ifamily, occupancy, day0, day1)
        if ediff > 0 and np.random.random() >= np.exp(-ediff/temp):
            continue
        else:
            move_family(ifamily, nfamily, day0, day1, assigned_day, family_on_day, occupancy)
            is_change = True
    
        # update etotal and check improvement
        etotal = etotal + ediff
        if etotal < etotal_low:
            print('{}/{}. Got improvement from {} to {}.'.format(iters, N_iters, etotal_low, etotal))
            dump_conf(assigned_day, path_dump_improved)
            etotal_low = etotal

# finalize
print('Final score:', cal_total(assigned_day, occupancy))
finalize(assigned_day, path_dump=path_dump_continue)
