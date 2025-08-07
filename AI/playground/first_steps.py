# ai imports
import torch
import torch.nn as nn
import torch.optim as optim

# graphing
import matplotlib.pyplot as plt

#import our dataset
import first_steps_data as bigdata

#other imports
import random as rand
import pandas as pd
import numpy as np

#---------------------- test algorithm -----------------------------

#saw this in a tutorial
#torch.manual_seed(34180) #for reproducibility? just picked a random number for now

# dataset!!
big_number = 20000 #modify this to 50k for higher accuracy? my laptops out of memory

#### HYPER PARAMETERS #####
SCALE_FACTOR = 500
HIDDEN_SIZE = 128
LAYERS = 1
EPOCHS = 150

#Other Variables
MODEL_NAME = "AbidLSTM"
MODEL_WEIGHT_PATH = 'weights'


testing_size = 50
dataset = bigdata.gen_data([], big_number)

test_set = bigdata.gen_data([], testing_size)
rand.shuffle(dataset) #turns out you have to have this as a single line


# create tensors (pytorch fancy arrays)
X = torch.tensor([[i/SCALE_FACTOR for i in seq] for seq, _ in dataset],
                  dtype=torch.float32) # first part, needs more looking into (vibe code black magic)

y = torch.tensor([target/SCALE_FACTOR for _, target in dataset],
                  dtype=torch.float32) # first part, needs more looking into (vibe code black magic)

#vibe code black magic plus abid tweakin
X = X.view(len(dataset), 5, 1) #adjust view since we have a 5 term sequence instead of 3
y = y.view(len(dataset), 1)


# Create our model class
class AbidLSTM(nn.Module):
    def __init__(self):
        super(AbidLSTM, self).__init__() # required because its a subclass
        #input size -> input a single list of numbers
        #hidden size -> number of neurons between the input layer and the output layer, "brain"
        #batch first -> idk

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            batch_first=True            

        )
        self.dropout = nn.Dropout(p=.2)
        self.linear = nn.Linear(HIDDEN_SIZE, 1)


    def forward(self, x):
        model_output, _ = self.lstm(x)
        last_time_step = model_output[:, -1, :]
        out = self.dropout(last_time_step)
        return self.linear(out)

# an instance of our baby model
chow = AbidLSTM()

# create our loss function and our optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(chow.parameters(), lr=0.01) #no clue about parameters() but our learning rate is 0.01

#learning rate decay?
#scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.5)

#time to train our model!
#loops over data set epoch times
loss_vals = []


# after new update, 
for epoch in range(EPOCHS):
    chow.train() #sets model to training mode

    optimizer.zero_grad() #resets gradients
    output = chow(X) #not really sure whats going on here
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    #scheduler.step()

    print(f"Epoch: {epoch} Loss:{loss.item():.4f}")
    loss_vals.append(loss.item())


# code to graph epochs vs loss
graph_x = [i for i in range(0, len(loss_vals))]
graph_y = [float(i) for i in loss_vals]

plt.figure(1)
plt.plot(graph_x, graph_y, color="b")

plt.title("Epochs vs Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")

plt.grid(True)




#testing our trained model

testers = bigdata.gen_testers(big_number)

#test_seq_un = torch.tensor([[7.0 / big_number, 8.0 / big_number, 9.0 / big_number]],
#                           dtype=torch.float32).view(1, 3, 1)
#
#test_seq_deux = torch.tensor([[40.0 / big_number, 41.0 / big_number, 42.0 / big_number]],
#                             dtype=torch.float32).view(1, 3, 1)
#
#test_seq_trois = torch.tensor([[50.0 / big_number, 51.0 / big_number, 52.0 / big_number]],
#                              dtype=torch.float32).view(1, 3, 1)

#visual representation of predicted vs actual (variables)
x_axis = [i for i in range(50)]
model_predictions = []
actual_values = []
percent_changes = []
differences = []

#switch to eval mode
chow.eval()
with torch.no_grad():
    i = 1
    print("TESTING!")
    for t_seq, actual in test_set:
    # for i in range(50):
        test_tensor = torch.tensor([[i / SCALE_FACTOR for i in t_seq]], dtype=torch.float32).view(1, 5, 1)

        #Old testing
        # uno = rand.randint(0, big_number)
        # test_seq = testers[uno]
        model_eval = chow(test_tensor)
        prediction = model_eval.item() * SCALE_FACTOR
        model_predictions.append(prediction)
        # actual = uno + 6 #adjusted for 5 term sequence
        actual_values.append(actual)

        #%change
        try: 
            change = 100*(abs(prediction - actual)/(actual))
        except ZeroDivisionError:
            print("Division by zero error")
        
        percent_changes.append(change)

        #difference
        difference = abs(prediction - actual)
        differences.append(difference)

        i += 1

datatable = {
    "Actual": actual_values,
    "Predicted": model_predictions,
    "%Change": percent_changes,
    "Difference": differences
}

dataframe = pd.DataFrame(datatable)
#dataframe.to_csv("AI\playground\datacsv", index=False)
print(dataframe)



#visual representation of predicted vs actual (matplotlib code)
plt.figure(2)
# Plot each set of Y-values
plt.scatter(x_axis, model_predictions, label="Predicted", color='r') #
plt.scatter(x_axis, actual_values, label="Actual", color='b')

#Draw line between both points (to visually assess accuracy)
for x, y_pred, y_actual in zip(x_axis, model_predictions, actual_values):
    plt.plot([x, x], [y_pred, y_actual], color='purple', linestyle='-', linewidth=1)

# Add labels and title
plt.xlabel('Tests')
plt.ylabel('Predictions and Actuals')
plt.title('Predicted and Actual Values per Test')

#add legend for better clarity
plt.legend()

plt.show()


#this is for saving the model (if its accurate)

#calculate the average for all the percent changes
MEAN_ERROR = np.mean(percent_changes)

#if the average error is lower than 1%, save the model
if (MEAN_ERROR <= 1.00):
    torch.save(chow.state_dict(), f"{MODEL_WEIGHT_PATH}/{MODEL_NAME}_{MEAN_ERROR * 100}.pth")
    print(f"{MODEL_WEIGHT_PATH}/{MODEL_NAME}_{MEAN_ERROR * 100}.pth")
