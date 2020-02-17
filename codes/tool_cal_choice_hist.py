import sys
from util_io import init, make_family_below_choice
from util_cost import cal_total

# params #
path_conf = '../input/another_pytorch_implementation.csv'

if len(sys.argv) == 2:
    path_conf = sys.argv[1]

assigned_day, family_on_day, occupancy = init(path_conf=path_conf)
family_on_choice, family_below_choice = make_family_below_choice(assigned_day)

print(*enumerate(family_on_choice), sep='\n')
print('\nCounts:')
print(*[(i, len(arr)) for i, arr in enumerate(family_on_choice)], sep='\n')
print('\nCounts for below (accumulative):')
print(*[(i, len(arr)) for i, arr in enumerate(family_below_choice)], sep='\n')
