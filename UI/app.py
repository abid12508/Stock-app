import tkinter as tk
from extraction import extract_symbol as es
import UI.search_suggestions as ss

# Initialize window
win = tk.Tk()
win.geometry("800x600")
win.title("Stock Predictor App")

# Dropdown/symbols/time intervals
time_period = ["INTRADAY", "DAILY", "WEEKLY", "MONTHLY"]  
symbols = es.symbols
time_intervals = [1, 5, 15, 30, 60]

# Time dropdown objects
tp_label = tk.Label(win, text="Time Period")
tp_sv = tk.StringVar(value="INTRADAY")  
tp_dropdown = tk.OptionMenu(win, tp_sv, *time_period)
#time dropdown placements
tp_label.place(relx=.2, rely=.1)
tp_dropdown.place(relx=.4, rely=.1)

# Company search objects
sb_label = tk.Label(win, text="Stock")
sb_sv = tk.StringVar()  
sb_entry = tk.Entry(win, textvariable=sb_sv)
# Company search placements
sb_label.place(relx=.2, rely=.2)
sb_entry.place(relx=.4, rely=.2)

# time intervals objects
ti_label = tk.Label(win, text="Time Intervals")
ti_iv = tk.StringVar(value="5")  
ti_dropdown = tk.OptionMenu(win, ti_iv, *time_intervals)
# time intervals placements
ti_label.place(relx=.2, rely=.3)
ti_dropdown.place(relx=.4, rely=.3)

# Initialize search suggestion
search_engine = ss.search_engine(win, sb_entry, sb_sv, symbols)

# Entry configuration: force uppercase and update suggestions
def on_entry_change(*args):
    # Force uppercase input
    current = sb_sv.get()
    sb_sv.set(current.upper())
    # Refresh suggestions
    search_engine.output()

sb_sv.trace_add("write", on_entry_change)

# Create function to call graph 
def show_graph_func():
    print("Add graphing soon...")
    print([tp_sv.get(), sb_sv.get(), ti_iv.get()])
    

# Create button to show graph
show_graph = tk.Button(win, text="Show Graph", command=show_graph_func)
show_graph.place(relx=.3, rely=.4)

# Make window mainloop
win.mainloop()
