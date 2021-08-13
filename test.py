import os
model = ['gcn', 'graphsage', 'dagnn', 'arma']
hidden = [8, 32, 128, 512]

for h in hidden:
    os.system("python3 gcn/train.py --dataset reddit --gpu 0 --n-hidden " + str(h))
    os.system("python3 graphsage/train_full.py --dataset reddit --gpu 0 --n-hidden " + str(h))
