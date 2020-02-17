import sys
from util_io import init

# params #
path_conf = '../input/another_pytorch_implementation.csv'

if len(sys.argv) == 2:
    path_conf = sys.argv[1]

assigned_day, family_on_day, occupancy = init(path_conf=path_conf)

hist = {}
for d in occupancy:
    hist[d] = hist.get(d,0)+1

print(*list(sorted(hist.items(), key=lambda x: x[0])), sep='\n')
