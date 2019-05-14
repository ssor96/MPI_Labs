import sys

data = zip(*[[float(x) for x in open(sys.argv[i]).readline().split()] for i in (1, 2)])

mx_diff = 0

for v1, v2 in data:
    mx_diff = max(mx_diff, abs(v1 - v2))
    if v1 != v1 or v2 != v2:
        raise ValueError('NAN presents in files!')


print('%.9f' % mx_diff)
