import torch
import random as rand

def gen_data(dataset, size):
    i = 1
    while i <= size:
        strt_pt = rand.randint(1, 1000)
        step = rand.randint(1, 25)
        direc = rand.choice([-1, 1])

        seq = [strt_pt + (j * step * direc) for j in range(3)]
        ans = seq[-1] + step * direc

        dataset.append((seq, ans))
        i += 1
    return dataset

def gen_testers(bigboy):

    data = []
    i = 1
    while i <= bigboy:
        data.append(torch.tensor([i / bigboy, (i+1) / bigboy, (i+2) / bigboy], dtype=torch.float32).view(1, 3, 1))
        i+=1
    return data
    


