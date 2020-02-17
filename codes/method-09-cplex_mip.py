"""

Final solution to Kaggle competition - Santa's Workshop Tour 2019

Mixed Integer Problem approach with CPLEX

Louis Yang 2020

**CPLEX installation guide:** After installing IBM ILOG CPLEX Optimization Studio locally using
the installer from IBM, for macOS, add following line to your `.bash_profile`:
```export PYTHONPATH="/Applications/CPLEX_Studio1210/cplex/python/3.6/x86-64_osx"```
For Windows, add the following to the front of `PATH` environment variable:
```C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\python\3.7\x64_win64\```
(Note the path will change if you are using different python or installing to a different folder)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from util_check import deep_check, check_valid_all
from util_cost import (
    cal_total, n_people, family_id_choice_to_pref_cost, cal_total_preference,
    cal_total_accounting, nd_ndp1_to_account_penality, family_id_days_to_pref_cost
)
from util_cost import choices as family_pref
from util_io import (
    init, finalize, dump_conf, assigned_day_to_family_on_day, assigned_day_to_occupancy
)

import cplex
print(cplex.__path__)  # make sure the path is the one we pick above
from docplex.mp.model import Model

## ====================  Parameters  ====================
# constants #
N_families = 5000
N_days = 100
N_min_people = 125
N_max_people = 300
# constants #

# params #
path_init_conf =     '../output/m24-improved-org.csv' # input solution
path_dump_improved = '../output/m24-improved-org.csv'    # output solution

num_cpu_cores = 12
time_limit = 9*60*60  # time limit in s for each loop (-1 for unlimited)

random_selected_days = False            # Random select mode or window sliding mode
N_selected_days = 0                     # N of selected days for expensive search
window_run = 1
new_init_update_frequency = 10          # Frequency of reloading external input solution 
occupancy_diff = 80                     # +- the occupancy of input solution for each day
occupancy_diff_low = 80                 # +- the occupancy of input solution for days with 125 occ
with_lowest_occupancy = True            # whether to add the 125 in the search occupancy for each day
max_family_rank = 6                     # maximum number of rank of the preference days for each family
use_hint = True                         # use current solution as initial solution for each loop
occupancy_count_as_variables = True     # use occupancy_counts as variable (seem faster)
redundant_occupancy_constraints = True  # use redundant constraints (faster)
all_family_constrains = True            # use sum of people from family equal sum from occupancy

target_pref_cost = 0               # Known best solution: 62868
target_pref_cost_error = 0
target_pref_cost_lower = 0
target_accounting_cost = 0         # Known best solution: 6020.043432
target_accounting_cost_error = 0
target_accounting_cost_lower = 0
max_accounting_cost_per_day = 300  # limit possible occupancy pairs with max accounting cost per day
min_choice_0_families = 3000       # minimum number of families that are at their choice 0
                                   # note: N_families - 62868 / 50 = 3742.64

## ====================  Predefined variables  ====================
families = range(N_families)
days = range(1, N_days + 1)
allowed_occupancy = range(N_min_people, N_max_people + 1)
possible_family_sizes = np.unique(n_people)
family_size_to_family_ids = {
    size: np.where(n_people == size)[0] for size in possible_family_sizes
}

# occupancy pairs [o, o_next] limited by accounting cost
viable_nd_ndp1 = nd_ndp1_to_account_penality <= max_accounting_cost_per_day

# Possible choice for the family
# last choice is any day that is not on the family's preferred days
N_choices_ori = family_id_choice_to_pref_cost.shape[1]  # 11
N_choices = min(N_choices_ori, max_family_rank)
N_family_pref = min(N_choices, N_choices_ori - 1)
print('Limit family choices:', N_choices_ori, '->', N_choices)
print('N of family preferred days:', N_family_pref)

# day to dictionary of families who choose this day with value as preference rank
days_family_prefered = [{} for day in range(N_days+1)]  # day = 0 should not be used
for family, pref in enumerate(family_pref):
    for rank, day in enumerate(pref):
        if rank < N_family_pref:
            days_family_prefered[day][family] = rank

## ====================  Functions  ====================
### Define search range
def build_expensive_days(assigned_day, i, p, changed_days):
    """ Define your logic for selecting N_selected_days of days to search all of
        the possible occupancy (125 - 300) in these days
    """
    if N_selected_days == 0:
        expensive_days = []
    elif random_selected_days:
        # randomly select + last changed days
        iw = int(2 * i / window_run)
        expensive_days = (
            list(np.random.choice(days, c + iw - len(changed_days), replace=False))
            + changed_days
        )
    else:
        # a sliding window of days in decreasing order of their preference cost
        families_cost = family_id_days_to_pref_cost[np.arange(N_families), assigned_day]
        df_family = pd.DataFrame({'day': assigned_day, 'cost': families_cost})
        day_pref_cost = df_family.groupby('day')[['cost']].sum()
        iw = int(i / N_selected_days)
        expensive_days = (
            day_pref_cost
            .sort_values('cost', ascending=False)
            .iloc[
                i%N_selected_days + p * (N_selected_days + iw) :
                i%N_selected_days + (p+1) * (N_selected_days + iw)
            ]
            .index.values
        )
    return expensive_days

def build_search_occupancy(occupancy, days, expensive_days, occupancy_diff, 
                           occupancy_diff_low):
    """ Construct occupancy search range for each day """
    search_occupancy = {}
    for day in days:
        if day in expensive_days:
            # Full search of occupancy for expensive days
            search_occupancy[day] = range(N_min_people, N_max_people+1)
        elif occupancy[day] == N_min_people:
            search_occupancy[day] = range(N_min_people, occupancy[day] + occupancy_diff_low + 1)
        else:
            # limit the occupancy choice to +- occupancy_diff of current solution    
            search_occupancy[day] = range(max(occupancy[day] - occupancy_diff, N_min_people),
                                          min(occupancy[day] + occupancy_diff, N_max_people) + 1)
        if with_lowest_occupancy:
            search_occupancy[day] = list(search_occupancy[day])
            if N_min_people not in search_occupancy[day]:
                search_occupancy[day] = [N_min_people] + search_occupancy[day]
    return search_occupancy

### Initialize MIP model
def init_mip_model(num_cpu_cores, time_limit):
    """ Initialize a new MIP model """
    solver = Model('')
    if num_cpu_cores > 0:
        solver.context.cplex_parameters.threads = num_cpu_cores
    print('Num treads:', solver.context.cplex_parameters.threads)
    if time_limit > 0:
        print('Set time limit:', time_limit)
        solver.set_time_limit(time_limit)
    solver.parameters.mip.tolerances.mipgap = 0  # set mip gap to 0
    return solver

### Variables
def build_occupancy_matrix_var(solver, search_occupancy, viable_nd_ndp1):
    """ Build occupancy_matrix variables as a dictionary to binary varibles.
        When occupancy_matrix[day, o, o_next] == True, it means the day has occupany == o, 
        and day + 1 has occupancy == o_next.
        For day < N_days, the arguments are [day, o, o_next], 
        where o and o_next is the current day occupancy and next day occupancy, respectively.
        For day == N_days, the arguments are [day, o], 
        since o == o_next for the last day.
        We further limit the occupancy to those on search_occupancy and viable_nd_ndp1. 
    """
    occupancy_keys_list = []
    for day in days:
        if day < N_days:
            for o in search_occupancy[day]:
                for o_next in search_occupancy[day + 1]:
                    if viable_nd_ndp1[o, o_next]:
                        occupancy_keys_list.append((day, o, o_next))
        else:
            # last day
            for o in search_occupancy[day]:
                if viable_nd_ndp1[o, o]:
                    occupancy_keys_list.append((day, o))
    return solver.binary_var_dict(occupancy_keys_list, name='o')

def build_occupancy_counts(solver, search_occupancy, assignment_matrix, unpreferred_day_counts):
    """ Build occupancy counts as int variables or build from assignment_matrix
        with constraints from search_occupancy
    """
    if occupancy_count_as_variables:
        # introduce intermedia int variables for occupancy
        lbs = [min(search_occupancy[day]) for day in days]
        ubs = [max(search_occupancy[day]) for day in days]
        occupancy_counts = solver.integer_var_dict(days, lb=lbs, ub=ubs, name='oc')

        for day in days:
            # find those family who like this day
            family_prefered = days_family_prefered[day]
            solver.add_constraint_(
                occupancy_counts[day] == (
                    solver.sum([
                        assignment_matrix[family, pref_rank] * n_people[family] 
                        for family, pref_rank in family_prefered.items()
                    ]) + (
                        solver.sum([
                            unpreferred_day_counts[day, size] * size
                            for size in possible_family_sizes
                        ]) if unpreferred_day_counts is not None else 0
                    )
                )
            )
    else:
        occupancy_counts = {}
        for day in days:
            # find those family who like this day
            family_prefered = days_family_prefered[day]
            occupancy_counts[day] = (
                solver.sum([
                    assignment_matrix[family, pref_rank] * n_people[family] 
                    for family, pref_rank in family_prefered.items()
                ]) + (
                    solver.sum([
                        unpreferred_day_counts[day, size] * size
                        for size in possible_family_sizes
                    ]) if unpreferred_day_counts is not None else 0
                )
            )
            solver.add_range(
                min(search_occupancy[day]), occupancy_counts[day], max(search_occupancy[day])
            )
    return occupancy_counts

### Constraints
def add_constraints(
    solver, assignment_matrix, unpreferred_day_counts, occupancy_matrix,
    search_occupancy, viable_nd_ndp1,
):
    # constraint 1: assignment_matrix normalization
    # each family only take one day (choice)
    solver.add_constraints_([
        solver.sum([assignment_matrix[family, c] for c in range(N_choices)]) == 1 
        for family in families
    ])

    # constraint 2: unpreferred day family count conservation for each family size
    if unpreferred_day_counts is not None:
        solver.add_constraints_([
            solver.sum([assignment_matrix[family, N_choices - 1]
                        for family in family_size_to_family_ids[size]])
            == solver.sum([unpreferred_day_counts[day, size] for day in days])
            for size in possible_family_sizes
        ])

    # constraint 3: occupancy_matrix normalization
    # each day only take 1 occupancy value
    for day in days:
        if day < N_days:
            occupancy_normalization = solver.sum([
                occupancy_matrix[day, o, o_next] 
                for o in search_occupancy[day]
                for o_next in search_occupancy[day + 1]
                if viable_nd_ndp1[o, o_next]
            ])
        else:
            occupancy_normalization = solver.sum([
                occupancy_matrix[day, o] 
                for o in search_occupancy[day]
                if viable_nd_ndp1[o, o]
            ])
        solver.add_constraint_(occupancy_normalization == 1)

    # constraint 4: link occupancy boolean matrix to occupancy counts
    for day in days:
        if day < N_days:
            sum_from_occupancy_matrix = solver.sum([
                occupancy_matrix[day, o, o_next] * o 
                for o in search_occupancy[day]
                for o_next in search_occupancy[day + 1]
                if viable_nd_ndp1[o, o_next]
            ])
        else:
            sum_from_occupancy_matrix = solver.sum([
                occupancy_matrix[day, o] * o 
                for o in search_occupancy[day]
                if viable_nd_ndp1[o, o]            
            ])
        solver.add_constraint_(occupancy_counts[day] == sum_from_occupancy_matrix)

    # constraint 5: next day occupancy consistency (similar to previous constraint)
    solver.add_constraints_([
        occupancy_counts[day + 1] == solver.sum([
            occupancy_matrix[day, o, o_next] * o_next
            for o in search_occupancy[day]
            for o_next in search_occupancy[day + 1]
            if viable_nd_ndp1[o, o_next]            
        ])
        for day in days if day < N_days
    ])

    # constraint 6: redudant constraints on occupancy_matrix
    # this constraint help MIP solver solve faster
    if redundant_occupancy_constraints:
        for day in days:
            if day + 1 < N_days:
                solver.add_constraints_([
                    solver.sum([
                        occupancy_matrix[day, o_other, o] 
                        for o_other in search_occupancy[day] if viable_nd_ndp1[o_other, o]
                    ]) == solver.sum([
                        occupancy_matrix[day + 1, o, o_other]
                        for o_other in search_occupancy[day + 2] if viable_nd_ndp1[o, o_other]
                    ])
                    for o in search_occupancy[day + 1]
                ])
        solver.add_constraints_([
            solver.sum([
                occupancy_matrix[N_days - 1, o_other, o] 
                for o_other in search_occupancy[N_days - 1] if viable_nd_ndp1[o_other, o]
            ]) == occupancy_matrix[N_days, o] if viable_nd_ndp1[o, o] else 0
            for o in search_occupancy[N_days]
        ])

    # constraint 7: family choices limit
    if min_choice_0_families > 0:
        solver.add_constraint_(
            solver.sum([assignment_matrix[family, 0] for family in families]) 
            >= min_choice_0_families
        )
    
    # constraint 8: require sum of family sizes equal sum of occupancy counts
    if all_family_constrains:
        print(n_people.sum(), type(n_people.sum()))
        solver.add_constraint_(n_people.sum() == solver.sum([occupancy_counts[d] for d in days])) 

### Cost functions
def build_family_pref_cost(
    solver, assignment_matrix, target_pref_cost=0, target_pref_cost_error=0,
    target_pref_cost_lower=0,
):
    """ Build total family preference cost from summing over assignment_matrix, and
        apply constraints if target prefence cost exists.
    """
    family_pref_cost = solver.sum([
        assignment_matrix[family, c] * family_id_choice_to_pref_cost[family, c]
        for family in families for c in range(1, N_choices)
    ])

    # preference cost constraints
    if target_pref_cost > 0:
        if target_pref_cost_error > 0:
            print('Limit preference cost in range')
            solver.add_range(
                target_pref_cost - target_pref_cost_error,
                family_pref_cost,
                target_pref_cost + target_pref_cost_error
            )
        else:
            print('Limit preference cost exactly')
            solver.add_constraint_(family_pref_cost == target_pref_cost)
    elif target_pref_cost_lower > 0:
        print('Set preference cost lower bound')
        solver.add_constraint_(family_pref_cost >= target_pref_cost_lower)        
    return family_pref_cost

def build_accounting_cost(
    solver, occupancy_matrix, search_occupancy, viable_nd_ndp1, target_accounting_cost=0,
    target_accounting_cost_error=0, target_accounting_cost_lower=0,
):
    """ Build total accounting cost from summing over occupancy_matrix, and
        apply constraints if target accounting cost exists.
    """

    accounting_cost = (
        solver.sum([
            occupancy_matrix[day, o, o_next] * nd_ndp1_to_account_penality[o, o_next]
            for day in days if day < N_days
            for o in search_occupancy[day] for o_next in search_occupancy[day + 1]
            if viable_nd_ndp1[o, o_next] and o > N_min_people
        ]) +
        solver.sum([
            occupancy_matrix[N_days, o] * nd_ndp1_to_account_penality[o, o]
            for o in search_occupancy[N_days]
            if viable_nd_ndp1[o, o] and o > N_min_people  
        ])
    )

    # accounting cost constraints
    if target_accounting_cost > 0 and target_accounting_cost_error > 0:
        print('Range limit accounting cost')
        solver.add_range(
            target_accounting_cost - target_accounting_cost_error,
            accounting_cost,
            target_accounting_cost + target_accounting_cost_error
        )
    elif target_accounting_cost_lower > 0:
        print('Lower bound accounting cost')
        solver.add_constraint_(accounting_cost >= target_accounting_cost_lower)
    return accounting_cost

### Initial solution
def build_init_solution(
    solver, assigned_day, occupancy, search_occupancy, viable_nd_ndp1,
    assignment_matrix, occupancy_matrix, occupancy_counts,
):
    """ Build initial hint solution for the MIP solver """
    from docplex.mp.solution import SolveSolution
    var_value_map = {}

    for family in families:
        for c in range(N_choices):
            var_value_map[assignment_matrix[family, c]] = float(
                assigned_day[family] == family_pref[family, c]
            )
    for day in days:
        if day < N_days:
            for o in search_occupancy[day]:
                for o_next in search_occupancy[day + 1]:
                    if viable_nd_ndp1[o, o_next]:
                        var_value_map[occupancy_matrix[day, o, o_next]] = float(
                            (occupancy[day] == o) and (occupancy[day + 1] == o_next)
                        )
                    else:
                        assert not ((occupancy[day] == o) and (occupancy[day + 1] == o_next)), \
                        'Hint not valid at (%i, %i, %i)'%(day, o, o_next)
    for o in search_occupancy[N_days]:
        if viable_nd_ndp1[o, o]:
            var_value_map[occupancy_matrix[N_days, o]] = float(occupancy[N_days] == o)
        else:
            assert not (occupancy[N_days] == o), \
            'Hint not valid at (%i, %i, %i)'%(N_days, o, o)

    if occupancy_count_as_variables:
        for day in days:
            var_value_map[occupancy_counts[day]] = float(occupancy[day])

    init_solution = SolveSolution(solver, var_value_map)
    return init_solution

### MIP progress listener
# See http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.progress.html#docplex.mp.
# progress.ProgressClock 
# for progress clock parameters meaning
from docplex.mp.progress import TextProgressListener, ProgressClock, SolutionRecorder

class MyProgressListener(SolutionRecorder):
    """ Progress listener to save intermediate improved solution """
    def __init__(self, initial_score=999999, clock=ProgressClock.Gap, absdiff=None, reldiff=None):
        super(MyProgressListener, self).__init__(clock, absdiff, reldiff)
        self.current_objective = initial_score
        
    def notify_solution(self, sol):
        if self.current_progress_data.current_objective >= self.current_objective:
            return
        print('Improved solution')
        super(MyProgressListener, self).notify_solution(sol)
        self.current_objective = self.current_progress_data.current_objective
        assigned_day_new_raw = np.ones(N_families, dtype='int32') * -1
        for family, choice in sol.get_value_dict(assignment_matrix, keep_zeros=False):
            assigned_day_new_raw[family] = (
                family_pref[family, choice] if choice < N_family_pref else -1
            )
        solution = pd.DataFrame(data=families, columns = ['family_id'])
        solution['assigned_day'] = assigned_day_new_raw
        solution.to_csv(path_dump_improved, index=False)
        
    def get_solutions(self):
        return self._solutions

### Get solution
def distribute_unpreferred_day(assigned_day, unpreferred_day_counts_sol, n_people):
    """ Distribute unpreferred day to each family who has -1 day assigned """
    assigned_day = assigned_day.copy()
    unpreferred_days = {size: [] for size in possible_family_sizes}
    for size in possible_family_sizes:
        for day, quota in enumerate(unpreferred_day_counts_sol[size]):
            unpreferred_days[size] = unpreferred_days[size] + [day] * quota
    unpreferred_day_headers = {size: 0 for size in possible_family_sizes}
    for family, (day, size) in enumerate(zip(assigned_day, n_people)):
        if day == -1:
            assigned_day[family] = unpreferred_days[size][unpreferred_day_headers[size]]
            unpreferred_day_headers[size] += 1
    return assigned_day

def extract_solution(sol, assignment_matrix, unpreferred_day_counts):
    """ Extract the solution of `assigned_day`, `family_on_day`, and `occupancy` from 
        the MIP solution object `sol`, and distribute unperferred days
    """
    # -1 reserves for unpreferred day
    assigned_day_new_raw = np.ones(N_families, dtype='int32') * -1
    for family, choice in sol.get_value_dict(assignment_matrix, keep_zeros=False):
        assigned_day_new_raw[family] = (
            family_pref[family, choice] if choice < N_family_pref else -1
        )

    if unpreferred_day_counts is not None:
        unpreferred_day_counts_sol_dict = sol.get_value_dict(unpreferred_day_counts)
        unpreferred_day_counts_sol = {
            size: [0]+[int(unpreferred_day_counts_sol_dict[day, size]) for day in days]
            for size in possible_family_sizes
        }
        print('Unpreferred families slots:')
        print({size: sum(counts) for size, counts in unpreferred_day_counts_sol.items()})

        assigned_day_new = distribute_unpreferred_day(
            assigned_day_new_raw, unpreferred_day_counts_sol, n_people
        )
        print('N family unpreferred assigned:', 
              (~(assigned_day_new == assigned_day_new_raw)).sum())
    else:
        assigned_day_new = assigned_day_new_raw

    family_on_day_new = assigned_day_to_family_on_day(assigned_day_new)
    occupancy_new = assigned_day_to_occupancy(assigned_day_new)
    return assigned_day_new, family_on_day_new, occupancy_new

def get_changed_days(occupancy_new, occupancy):
    occupancy_change = (occupancy_new != occupancy)[1:-1]
    return list(np.array(days)[occupancy_change])

## Load input solution
assigned_day, family_on_day, occupancy = init(path_conf=path_init_conf)
print('Init config:')
try:
    is_valid = deep_check(assigned_day, family_on_day, occupancy)
except:
    is_valid = False
initial_score = cal_total(assigned_day, occupancy)
print('Valid solution: ', is_valid)
print('Total score:    ', initial_score)
print('Preference cost:', cal_total_preference(assigned_day))
print('Accounting cost:', cal_total_accounting(occupancy))

best_score = initial_score


## ====================  Main Loop  ====================
N_batchs = int(N_days / N_selected_days) + 1 if N_selected_days > 0 else 1
k = 0
changed_days = []
for i, p in product(range(window_run), range(N_batchs)):
    if k % new_init_update_frequency == 0:
        # update initial solution 
        # (might be generated by other run of the similar program on other computer)
        assigned_day_o, family_on_day_o, occupancy_o = init(path_conf=path_init_conf)
        print('Init config:', k)
        try:
            is_valid = deep_check(assigned_day_o, family_on_day_o, occupancy_o)
        except:
            is_valid = False
        new_initial_score = cal_total(assigned_day_o, occupancy_o)
        if new_initial_score < best_score and is_valid:
            print('Total score:    ', new_initial_score)
            print('Preference cost:', cal_total_preference(assigned_day_o))
            print('Accounting cost:', cal_total_accounting(occupancy_o))
            print('using new conf')
            best_score = new_initial_score
            assigned_day = assigned_day_o
            family_on_day = family_on_day_o
            occupancy = occupancy_o
    k += 1

    # Update search_occupancy
    expensive_days = build_expensive_days(assigned_day, i, p, changed_days)
    print('[', i, ',', p,']', expensive_days)
    if len(expensive_days) == 0 and N_selected_days > 0:
        continue
    
    search_occupancy = build_search_occupancy(occupancy, days, expensive_days, occupancy_diff, 
                                              occupancy_diff_low)

    # ==== DOCplex model ====
    
    solver = init_mip_model(num_cpu_cores, time_limit)

    ## --- Variables ---
    # assignment matrix [family, pref_rank]
    assignment_matrix = solver.binary_var_matrix(families, range(N_choices), 'x')

    # unpreferred_day_counts [day, size]
    if N_choices_ori <= N_choices:
        print('using unpreferred day counts')
        ub = int(N_max_people / possible_family_sizes.min())
        unpreferred_day_counts = solver.integer_var_matrix(
            days, possible_family_sizes, lb=0, ub=ub, name='d'
        )
        print(len(unpreferred_day_counts))
    else:
        unpreferred_day_counts = None

    # occupancy matrix [day, o, o_next]
    occupancy_matrix = build_occupancy_matrix_var(solver, search_occupancy, viable_nd_ndp1)
    
    ## --- Intermediate variables ---
    # Build occupancy counts from assignment_matrix and add
    # with constraint: each day can only have 125-300 people
    occupancy_counts = build_occupancy_counts(solver, search_occupancy, assignment_matrix, 
                                              unpreferred_day_counts)
    
    ## --- Constraints ---
    add_constraints(
        solver, assignment_matrix, unpreferred_day_counts, occupancy_matrix,
        search_occupancy, viable_nd_ndp1,
    )    

    ### Family preference cost
    family_pref_cost = build_family_pref_cost(
        solver, assignment_matrix, target_pref_cost, target_pref_cost_error,
        target_pref_cost_lower,
    )

    ### Accounting cost
    accounting_cost = build_accounting_cost(
        solver, occupancy_matrix, search_occupancy, viable_nd_ndp1, target_accounting_cost,
        target_accounting_cost_error, target_accounting_cost_lower,
    )

    ## --- Objective ---
    solver.minimize(family_pref_cost + accounting_cost)
    
    # ==== Initial solution ====
    if use_hint:
        print('Use hint solution!')
        init_solution = build_init_solution(
            solver, assigned_day, occupancy, search_occupancy, viable_nd_ndp1,
            assignment_matrix, occupancy_matrix, occupancy_counts,
        )
        solver.add_mip_start(init_solution)

    # ==== Solve ====

    # Save intermediate improved solution
    my_progress_listener = MyProgressListener(initial_score=best_score, 
                                              clock=ProgressClock.Objective)  
    solver.add_progress_listener(my_progress_listener)  


    print('N of variables (binary, int):', solver.number_of_variables, 
          '(', solver.number_of_binary_variables, ',', solver.number_of_integer_variables, ')')
    print('N of constraints:', solver.number_of_constraints)
    print('Time limit:', solver.get_time_limit())

    ## --- Solve ---
    sol = solver.solve(log_output=True)

    if sol is None:
        sol = my_progress_listener.get_solutions()[-1]
    print('Solution status:', solver.get_solve_status())
    print('Total cost:', sol.objective_value, sol.get_objective_value())
    print('Time:', '%.3f' % solver.get_solve_details().time, 's')

    # ==== Get Solution ====
    assigned_day_new, family_on_day_new, occupancy_new = extract_solution(
        sol, assignment_matrix, unpreferred_day_counts
    )
    try:
        is_valid = deep_check(assigned_day_new, family_on_day_new, occupancy_new)
    except:
        is_valid = False
    new_score = cal_total(assigned_day_new, occupancy_new)
    print('Valid solution: ', is_valid)
    print('Total score:    ', new_score, '(', new_score - best_score, ')')
    print('Preference cost:', cal_total_preference(assigned_day_new))
    print('Accounting cost:', cal_total_accounting(occupancy_new))
    
    solver.end()
    
    ## --- update ---
    if new_score < best_score:
        changed_days = get_changed_days(occupancy_new, occupancy)  # search them next times
        
        dump_conf(assigned_day_new, path_dump_improved)
        
        best_score = new_score
        assigned_day = assigned_day_new
        family_on_day = family_on_day_new
        occupancy = occupancy_new

print('Total score change:', best_score - initial_score)


## ====================  Final Output  ====================

is_improved = new_score < initial_score
if is_valid and (is_improved or (path_dump_improved != path_init_conf)):
    print('output to', path_dump_improved)
    dump_conf(assigned_day_new, path_dump_improved)

plt.plot()
