
import os

path = './logs'

for d in sorted(os.listdir(path)):
    if d.endswith('0.log'):
        print()
        bm, nm = 0, 0
    p = os.path.join(path, d)
    with open(p, 'r') as f:
        lines = f.read().split('\n')
    for l in lines:
        if l.startswith('Average of base mIoU'):
            t = l.split()
            bm += float(t[4])
            nm += float(t[9])
            print(f'{d:32s}base mIoU {t[4]}\tnovel mIoU {t[9]}')
    if d.endswith('3.log'):
        print(f'{"":32s}     mean {bm/4:.4f}\t      mean {nm/4:.4f}')