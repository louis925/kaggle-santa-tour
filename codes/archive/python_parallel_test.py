from multiprocessing import Pool

ss= 111

def aaa(a, ss=ss, sss=sss):
    return ss*a+22, sum(sss)

sss= [1,2,3]

with Pool(processes=2) as p:
    results = p.map(aaa, [1, 2])

print(results)
