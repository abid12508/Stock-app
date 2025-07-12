from sklearn.linear_model import LinearRegression
import numpy as np
from collections import defaultdict

class Reg:
    def __init__(self, data):
        self.data = data

        self.close_prices = self.data["Close"]
        self.close_prices_list = []
        self.dates = self.data.index.tolist()
        self.date_dict = defaultdict()

        self.x = None
        self.y = None

        self.predicted_list = []
    
    def column_restruct(self): 

        for i in self.close_prices.values:
            self.close_prices_list.append(float(i))

        for i, e in enumerate(self.dates):
            self.date_dict[i] = e
            self.dates[i] = i

    def calc_reg(self):
        # Example data
        x = self.dates       # Independent variable
        y = self.close_prices_list      # Dependent variable

        # Reshape x for sklearn
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x, y)

        last_day = int(x[-1][0])  # Get the last index number (e.g., 251)
        future_days = np.array([[last_day], [last_day + 1]])

        y_pred = model.predict(future_days)
        # Convert predictions to list
        self.predicted_list = y_pred.tolist()

        self.x = x
        self.y = y

    def output(self):
        print([
            f"Last y value: {repr(self.y[-1])}",
            f"Next y prediction: {repr(self.predicted_list[-1])}"
        ])


    def do_all(self):
        self.column_restruct()
        self.calc_reg()
        self.output()

