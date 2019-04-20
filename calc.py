#%%
# Average Precision, Recall and F1 score.
r = 0
p = 0
ta = 0
c = 0
f = open('output2.txt', 'r')
for l in f.readlines()[1:]:
    t = l.split()
    r += float(t[6].split(':')[1].strip())
    p += float(t[7].split(':')[1].strip())
    ta += float(t[-1].split(':')[1].strip())
    c += 1

r = r / c
p = p / c
ta = ta / c
f1 = 2 * p * r / (p + r)
print('P, R, TA, F1')
print(p, r, ta, f1)