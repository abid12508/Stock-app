import torch
import random as rand

def gen_data(dataset, size):
    i = 1
    while i <= size:
        # strt_pt = rand.randint(1, 1000)
        step = rand.randint(1, 20)
        
        # attempts to be inclusive for all numbers
        if i % 2 == 1:
             strt_pt = rand.uniform(0, 500)    # positive start
        else:
             strt_pt = rand.uniform(0, 50)    # around zero
        
        #make sequences longer than 3 to train larger contexts
        
        seq = [strt_pt + (j * step) for j in range(5)]
        ans = seq[-1] + step

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

#AI implementation
""" Convert closing values into a list 
    Check Every 4 closing values (starting from the beginning of list)
    make first 3 into X tensor, 4th is answer (Y tensor).
    Activate LSTM to predict values, check difference, make adjustments 
    Shift tensor values by 1 (sliding window technique)
    Stop LSTM once future price is predicted
"""