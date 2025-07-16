import torch
import random as rand

def gen_data(dataset, size):
    i = 1
    while i <= size:
        strt_pt = rand.randint(1, 1000)
        step = rand.randint(1, 25)
        direc = rand.choice([-1, 1])

        #make sequences longer than 3 to train larger contexts
        seq = [strt_pt + (j * step * direc) for j in range(5)]
        ans = seq[-1] + step * direc

        dataset.append((seq, ans))
        i += 1
    return dataset

def gen_testers(bigboy):

    data = []
    i = 1
    while i <= bigboy:
        # adjusting this to a sequence of 5
        data.append(torch.tensor([i / bigboy, (i+1) / bigboy, (i+2) / bigboy, (i+3) / bigboy, (i+4) / bigboy], 
                                 dtype=torch.float32).view(1, 5, 1)) #adjust middle number to account for 5 term sequence
        i+=1
    return data
    


