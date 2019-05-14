from random import random, randint
import struct

max_mod = 10

def gen_rand_num():
    return (random() * 2 - 1) * max_mod


n = randint(10000, 20000)
k = 4 * 10**7#randint(n, n)

sm = [random() for _ in range(n)]
elements = dict()

for _ in range(k):
    r = 0
    c = 0
    while r == c or (r, c) in elements:
        r = randint(0, n - 1)
        c = randint(0, n - 1)
    v = gen_rand_num()
    elements[(r, c)] = v
    sm[r] += abs(v)

for i in range(n):
    sm[i] *= 1 if random() > 0.5 else -1
    elements[(i, i)] = sm[i]

x = [gen_rand_num() for _ in range(n)]

b = [0] * n

for (r, c), v in elements.items():
    b[r] += v * x[c]


with open('original.txt', 'w') as orig:
    print(n, file=orig)
    print(len(elements), file=orig)
    for (r, c), v in sorted(elements.items()):
        print(r, c, v, file=orig)

    print(*b, file=orig)

# for (r, c) in elements:
#     elements[(r, c)] /= -sm[r]
#
# for i in range(n):
#     b[i] /= sm[i]
#     del elements[(i, i)]

with open('test.bin', 'wb') as test:
    test.write(struct.pack('ii', n, len(elements)))
    for (r, c), v in sorted(elements.items()):
        test.write(struct.pack('iid', r, c, v))

    test.write(struct.pack('d' * n, *b))


with open('ans.txt', 'w') as ans:
    print(*x, file=ans)
