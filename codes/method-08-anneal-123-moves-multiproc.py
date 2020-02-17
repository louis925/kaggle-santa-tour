"""
"""

import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

from util_io import init, finalize, dump_conf, make_family_below_choice
from util_cost import cal_total, cal_diff_n
from util_cost import n_people
from util_check import check_valid_all, check_valid_move_n, deep_check
from util_events import move_n_family, correct_invalid

# constants #
N_families = 5000
N_days = 100
# constants #

# params #
path_init_conf =     '../input/another_pytorch_implementation.csv'
path_dump_improved = '../output/m08-improved.csv' # lowest cost

### test ###
N_runs = 3
list_temp = [100,10,3]
list_N_moved = [3,2,1]
list_upper_occupancy = [300,300,300]
list_N_iters = [100000,100000,100000]
### test ###

#N_runs = 100
#list_temp =    [0.1*t for i in range(10) for t in range(30, 20, -1)]
#list_N_moved = [1 if (i//10)%5==1 or (i//10)%5==3 else 2 for i in range(100)]

#list_upper_occupancy = [300+j for j in range(10, 0, -1) for i in range(4)] + [300 for i in range(60)]
#list_upper_occupancy = [300 for i in range(N_runs)]
#list_N_iters =         [10000000 for i in range(N_runs)]

### pick3 params ###
is_loop_move = True
level_choice_sampling = 1 # None for turn off choice sampling
### pick3 params ###

assert len(list_temp) == N_runs, 'size list_temp != N_runs: %d, %d' % (len(list_temp), N_runs)
assert len(list_N_moved) == N_runs, 'size list_N_moved != N_runs: %d, %d' % (len(list_N_moved), N_runs)
assert len(list_upper_occupancy) == N_runs, 'size list_upper_occupancy != N_runs: %d, %d' % (len(list_upper_occupancy), N_runs)
assert len(list_N_iters) == N_runs, 'size list_N_iters != N_runs: %d, %d' % (len(list_N_iters), N_runs)

N_dumps = 5000000 # dump every iters
# params #

# functions #
def determine_acceptance(ifamilies, nfamilies, day0s, day1s, occupancy, temp, upper_occupancy=300, zero_temp=False):
    # check validity
    if not check_valid_move_n(occupancy, day0s, day1s, nfamilies, upper_occupancy=upper_occupancy):
        return None

    # determine acceptance
    ediff = cal_diff_n(ifamilies, occupancy, day0s, day1s, nfamilies)
    if ediff <= 0:
        return ediff
    elif not zero_temp and np.random.random() < np.exp(-ediff/temp):
        return ediff
    else:
        return None

def pick1(assigned_day, family_below_choice):
    ifamilies, nfamilies, day0s, day1s = [np.array([-1]) for i in range(4)]
    
    # pick family
    ifamilies[0] = np.random.randint(N_families)
    nfamilies[0] = n_people[ifamilies]

    # pick day
    day0s[0] = assigned_day[ifamilies]
    day1s[0] = np.random.randint(1, N_days+1)

    return ifamilies, nfamilies, day0s, day1s

def pick2(assigned_day, family_below_choice):
    ifamilies, nfamilies, day0s, day1s = [np.array([-1, -1]) for i in range(4)]

    # pick families (always move 1 to 2)
    ifamilies[0] = np.random.randint(N_families)
    ifamilies[1] = np.random.randint(N_families)
    while assigned_day[ifamilies[0]] == assigned_day[ifamilies[1]]:
        ifamilies[1] = np.random.randint(N_families)
    nfamilies[0] = n_people[ifamilies[0]]
    nfamilies[1] = n_people[ifamilies[1]]

    # pick days
    day0s[0] = assigned_day[ifamilies[0]]
    day0s[1] = assigned_day[ifamilies[1]]
    day1s[0] = assigned_day[ifamilies[1]]
    day21 = np.random.randint(1, N_days)
    day21 = day21 if day21 < assigned_day[ifamilies[1]] else day21 + 1
    day1s[1] = day21

    return ifamilies, nfamilies, day0s, day1s

def pick3(assigned_day, family_below_choice, is_loop_move=is_loop_move, level_choice_sampling=level_choice_sampling):
    ifamilies, nfamilies, day0s, day1s = [np.array([-1, -1, -1]) for i in range(4)]

    # pick families
    a = np.random.choice(family_below_choice[level_choice_sampling])
    if level_choice_sampling is None:
        ifamilies[0] = np.random.randint(N_families)
    else:
        ifamilies[0] = np.random.choice(family_below_choice[level_choice_sampling])
    day0s[0] = assigned_day[ifamilies[0]]

    ifamilies[1] = -1
    while -1 == ifamilies[1] or assigned_day[ifamilies[1]] == day0s[0]:
        if level_choice_sampling is None:
            ifamilies[1] = np.random.randint(N_families)
        else:
            ifamilies[1] = np.random.choice(family_below_choice[level_choice_sampling])
    day0s[1] = assigned_day[ifamilies[1]]

    ifamilies[2] = -1
    while -1 == ifamilies[2] or assigned_day[ifamilies[2]] == day0s[0] or assigned_day[ifamilies[2]] == day0s[1]:
        if level_choice_sampling is None:
            ifamilies[2] = np.random.randint(N_families)
        else:
            ifamilies[2] = np.random.choice(family_below_choice[level_choice_sampling])
    day0s[2] = assigned_day[ifamilies[2]]
        
    # nfamilies
    nfamilies[0] = n_people[ifamilies[0]]
    nfamilies[1] = n_people[ifamilies[1]]
    nfamilies[2] = n_people[ifamilies[2]]

    # pick day1s: see method-07 for is_loop_move and not methods
    if is_loop_move:
        day1s[0] = day0s[1]
        day1s[1] = day0s[2]
        day1s[2] = day0s[0]
    else:
        day1s[0] = day0s[1] if np.random.random() < 0.5 else day0s[2]
        day1s[1] = day0s[2]
        day1s[2] = np.random.randint(1, N_days)
        day1s[2] = day1s[2] if day1s[2] < day0s[2] else day1s[2] + 1
    
    return ifamilies, nfamilies, day0s, day1s


pick_func = {1: pick1, 2: pick2, 3: pick3}

def run_search(args):
    """ running the actual search loop. 
    if run in multi-processing, it is a run for a processor
    """
    temp, upper_occupancy, N_moved, N_iters, assigned_day, family_on_day, occupancy, family_below_choice = args
    
    # init vars
    events = []
    n_improved = 0 # indicate numbers of events until the improved score
    ediff_whole = 0
    ediff_low = 0

    for iters in range(N_iters):
        ifamilies, nfamilies, day0s, day1s = pick_func[N_moved](assigned_day, family_below_choice)
        ediff = determine_acceptance(ifamilies, nfamilies, day0s, day1s, occupancy, temp, upper_occupancy=upper_occupancy)
        if ediff is None:
            continue
        
        events.append( [ifamilies, nfamilies, day0s, day1s] )
        move_n_family(ifamilies, nfamilies, day0s, day1s, assigned_day, family_on_day, occupancy)

        # update etotal and check improvement
        ediff_whole += ediff
        if ediff_whole < ediff_low:
            n_improved = len(events)
            ediff_low = ediff_whole

    return ediff_low, n_improved, events
# functions #


### !!!!! ----------------------------- !!!!! ###


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: python3 %s N_workers.' % sys.argv[0]
    Nproc = int(sys.argv[1])

    # init
    print('list_temp:')
    print(list_temp)
    print('list_N_moved:')
    print(list_N_moved)
    print('list_upper_occupancy:')
    print(list_upper_occupancy)
    print('list_N_iters:')
    print(list_N_iters)

    assigned_day, family_on_day, occupancy = init(path_conf=path_init_conf)
    family_on_choice, family_below_choice = make_family_below_choice(assigned_day)
    etotal_low = cal_total(assigned_day, occupancy)
    print('Init config cost:', etotal_low)

    for i_run in range(N_runs):
        # params set up
        temp = list_temp[i_run]
        upper_occupancy = list_upper_occupancy[i_run]
        N_moved = list_N_moved[i_run]
        N_iters = list_N_iters[i_run]

        # correct invalid days (if > upper_occ)
        correct_invalid(assigned_day, family_on_day, occupancy, upper_occupancy=upper_occupancy)
        deep_check(assigned_day, family_on_day, occupancy, upper_occupancy=upper_occupancy)

        # init vars
        is_change = False
   
        etotal_low = cal_total(assigned_day, occupancy)
        print('At temp = %.1f, upper_occ = %d, N_moved = %d, N_iters = %d. init cost: %.5f' % \
              (temp, upper_occupancy, N_moved, N_iters, etotal_low))

        # perform search
        if Nproc == 1:
            ediff_low, n_improved, events = run_search( (temp, upper_occupancy, N_moved, N_iters, \
                                                         assigned_day, family_on_day, occupancy, family_below_choice) )
            if ediff_low < -1e-10:
                etotal_low += ediff_low

                # revert last few events to get improved conf
                for ifamilies, nfamilies, day0s, day1s in events[:n_improved-1:-1]:  # day1s -> day0s
                    move_n_family(ifamilies, nfamilies, day1s, day0s, assigned_day, family_on_day, occupancy)

                print('{}/{}. Got improvement from {} to {}.'.format(i_run, N_runs, etotal_low - ediff_low, etotal_low))
                dump_conf(assigned_day, path_dump_improved)

                # forward back events
                for ifamilies, nfamilies, day0s, day1s in events[:n_improved-1:-1]:  # day0s -> day1s
                    move_n_family(ifamilies, nfamilies, day0s, day1s, assigned_day, family_on_day, occupancy)

        else:
            with Pool(processes=Nproc) as p:
                results = p.map(run_search, [(temp, upper_occupancy, N_moved, N_iters, \
                                              assigned_day, family_on_day, occupancy, family_below_choice) for i in range(Nproc)])

            results.sort(key= lambda x: x[0])

            # if cost improved, update to lowest conf found first 
            if results[0][0] < -1e-10:
                for ifamilies, nfamilies, day0s, day1s in results[0][2][:results[0][1]]:
                    move_n_family(ifamilies, nfamilies, day0s, day1s, assigned_day, family_on_day, occupancy)
                etotal_low += results[0][0]
                print('{}/{}. Got improvement from {} to {}.'.format(i_run, N_runs, etotal_low - results[0][0], etotal_low))
                dump_conf(assigned_day, path_dump_improved)

            # then found if there are additional treat (improves) by checking events 1 by 1
            for ediff_low, n_improved, events in results:
                for ifamilies, nfamilies, _, day1s in events:
                    day0s = np.array([assigned_day[i] for i in ifamilies])
                    if (day0s == day1s).all():
                        continue

                    ediff = determine_acceptance(ifamilies, nfamilies, day0s, day1s, occupancy, 0, \
                                                 upper_occupancy=upper_occupancy, zero_temp=True)
                    if ediff is not None:
                        move_n_family(ifamilies, nfamilies, day0s, day1s, assigned_day, family_on_day, occupancy)
                        etotal_low += ediff
                        print('{}/{}. Got improvement from {} to {}. (additional treat)'\
                              .format(i_run, N_runs, etotal_low - ediff, etotal_low))
                        dump_conf(assigned_day, path_dump_improved)

    # finalize
    print('Final score:', cal_total(assigned_day, occupancy))












