import numpy as np
import pandas as pd

from util_cost import cal_total
from util_cost import n_people, choices
from util_check import check_valid_all

# constants #
N_families = 5000
N_days = 100
N_choices = 10
# constants #

def assigned_day_to_family_on_day(assigned_day):
    family_on_day = [set() for _ in range(N_days+1)] # 0 is empty set
    for i, day in enumerate(assigned_day):
        family_on_day[day].add(i)
    return family_on_day

def assigned_day_to_occupancy(assigned_day):
    occupancy = np.zeros(N_days+2, dtype='int32') # 0 is 0
    for i, n in enumerate(n_people):
        occupancy[assigned_day[i]] += n
    occupancy[0] = 125
    occupancy[-1] = occupancy[-2]
    return occupancy

def read_conf(path):
    '''
    Return
        assigned_day: 1-D nparray (index: family_id, value: assigned_day)
        family_on_day: list of sets (list index: day_id, set value: family_id)
        occupancy: 1-D nparray (index: day_id, value: N of people)
    '''
    df = pd.read_csv(path)
    df = df.sort_values(by='family_id')

    # assigned_day
    assigned_day = df['assigned_day'].astype('int32').values

    # family_on_day
    family_on_day = [set() for _ in range(N_days+1)] # 0 is empty set
    for i, day in enumerate(assigned_day):
        family_on_day[day].add(i)

    # occupancy
    occupancy = np.zeros(N_days+2, dtype='int32') # 0 is 0
    for i, n in enumerate(n_people):
        occupancy[assigned_day[i]] += n
    occupancy[0] = 125
    occupancy[-1] = occupancy[-2]

    if not check_valid_all(occupancy):
        ans = input('Config in {} not valid. Continue? (Y/N) '.format(path))
        if ans != 'Y':
            raise Exception('Abort.')

    print('Read config completed.')
    return assigned_day, family_on_day, occupancy

def dump_conf(assigned_day, path):
    pd.DataFrame(
        {'family_id': range(N_families), 'assigned_day': assigned_day}
    ).to_csv(path, index=False)

def make_family_below_choice(assigned_day):
    family_on_choice = [[ifa for ifa in range(N_families) if assigned_day[ifa]==choices[ifa,ich]] for ich in range(N_choices)]
    
    not_in_choice = set(range(N_families))
    for ich in range(N_choices):
        not_in_choice = not_in_choice - set(family_on_choice[ich])
    family_on_choice.append( list(not_in_choice) )

    family_below_choice = [family_on_choice[-1]]
    for ich in range(N_choices-1, -1, -1):
        family_below_choice.append(family_below_choice[-1]+family_on_choice[ich])
    family_below_choice = family_below_choice[::-1]

    return family_on_choice, family_below_choice
    

# ****************************************** #

def init(path_conf=None):
    print('Read initial configs...')
    assigned_day, family_on_day, occupancy = read_conf(path_conf)

    return assigned_day, family_on_day, occupancy

def finalize(assigned_day, path_dump=None):
    if path_dump is not None:
        print('Save assigned day to {}'.format(path_dump))
        dump_conf(assigned_day, path_dump)
    else:
        print('Not save assigned_day to file.')
