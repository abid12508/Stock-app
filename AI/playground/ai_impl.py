# our model loading import
import pandas as pd
import torch

# our ai traning file which contains our model object
print("before importing first steps")
from first_steps import AbidLSTM
print("after importing and before fsdata")
import first_steps_data 
print("after both imports")
#SCALE FACTOR FROM MODEL!
SCALE_FACTOR = 579.783377734821

#brains of the model
print("Before loading model weight")
model_weight = torch.load("./weights/AbidLSTM_0.09475_579.783377734821_2025-08-07.pth")
print("after loading model weights")

model = AbidLSTM()
print("created model object")

print("loading model weights into model object")
model.load_state_dict(model_weight)

#set our model to evaluation mode
print("setting model to evaluation mode")
model.eval()

#generate some testers
test_set = first_steps_data.gen_data([], 10)

model_predictions = []
actual_values = []
percent_changes = []
differences = []

i = 0
for t_seq, actual in test_set:
        test_tensor = torch.tensor([[i / SCALE_FACTOR for i in t_seq]], dtype=torch.float32).view(1, 5, 1)

        model_eval = model(test_tensor)
        prediction = model_eval.item() * SCALE_FACTOR
        
        model_predictions.append(prediction)
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