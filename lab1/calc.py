_, t1 = input().split()

t1 = float(t1)
print('({};{})'.format(1, 1))
while True:
    n, t = input().split()
    n = int(n)
    t = float(t)
    print('({};{})'.format(n, (t1 / t) / n))
    if n == 36:
        break