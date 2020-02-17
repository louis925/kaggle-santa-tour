import sys
from util_io import init
from util_cost import cal_total

# params #
path_conf1 = '../input/another_pytorch_implementation.csv'
path_conf2 = '../input/another_pytorch_implementation.csv'

if len(sys.argv) == 3:
    path_conf1 = sys.argv[1]
    path_conf2 = sys.argv[2]


assigned_day1, family_on_day1, occupancy1 = init(path_conf=path_conf1)
assigned_day2, family_on_day2, occupancy2 = init(path_conf=path_conf2)

diffs = []
for n1, n2 in zip(occupancy1, occupancy2):
    diffs.append(n1 - n2)

hist = {}
for d in diffs:
    hist[d] = hist.get(d,0)+1

print(*list(sorted(hist.items(), key=lambda x: x[0])), sep='\n')
