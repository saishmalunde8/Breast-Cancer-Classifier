from model import *
from dataset import *
import tkinter as tk
import pandas as pd

HEIGHT = 500
WIDTH = 600

parameters = None
label_file_explorer = None

# ------------------------------------------------------------------------------------------------------------------------------

# Button functions

def evaluate_model():
    accuracy_val = evaluate(X_val_sc, y_val, parameters)

def evaluate_val(placeholder):
    X_val_sc, y_val = get_val_dataset()
    accuracy_val = evaluate(X_val_sc, y_val, parameters)
    make_label(placeholder, f"Accuracy on Validation dataset: {round(accuracy_val, 2)} %", 40, 0.5, 0.3, 0.8, 0.1, 'n')

def evaluate_test(placeholder):
    X_test_sc,  y_test = get_test_dataset()
    accuracy_test = evaluate(X_test_sc, y_test, parameters)
    make_label(placeholder, f"Accuracy on Test dataset: {round(accuracy_test, 2)} %", 40, 0.5, 0.7, 0.8, 0.1, 'n')

def browseFiles(placeholder): 
    global label_file_explorer

    filename = tk.filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("csv", "*.csv*"), ("all files", "*.*"))) 
       
    # Change label contents 
    label_file_explorer.configure(text="File Opened: "+filename)

    x_input = pd.read_csv(filename).values[:, -1]
    x_input = x_input.reshape(x_input.shape[0], 1)

    print("Input feature: shape", x_input.shape)
    make_button(placeholder, "Predict", 40, lambda:predict(x_input, parameters), 0.5, 0.6, 0.3, 0.1,'n')

    # test()

def get_train_model_entries(layer_dims, learning_rate, iterations):
    global parameters

    X_train_sc, y_train = get_train_dataset()

    layer_dims = list(map(int, (layer_dims.split(', '))))
    learning_rate = float(learning_rate)
    iterations = int(iterations)

    # print(X_train_sc, y_train)

    parameters = L_layer_model(X_train_sc, y_train, layer_dims, print_cost = True)

    tk.messagebox.showinfo("Task Completed", "Model has trained.\nParameters of the model are saved.\nYou can see the cost-iterations graph now!" )

    # print(layer_dims, learning_rate, iterations)

# UI
def make_file_browser(placeholder, _relx, _rely,_anchor, _font):
    global label_file_explorer

    label_file_explorer = tk.Label(placeholder,  text = "No file choosen yet", width = 100, height = 4,  fg = "blue", font = _font) 
    label_file_explorer.place(relx=_relx, rely= _rely, anchor = _anchor)

    button_explore = tk.Button(placeholder, text = "Browse Files", command = lambda: browseFiles(placeholder), font = _font)
    button_explore.place(relx=_relx, rely= _rely + 0.2, anchor = _anchor)

def make_button(placeholder, _text, _font, _command, _relx, _rely, _relwidth, _relheight, _anchor):
    button = tk.Button(placeholder, text=_text, font=40, command=_command)
    button.place(relx=_relx, rely= _rely, relwidth=_relwidth, relheight=_relheight, anchor = _anchor)

def make_text(placeholder, _text, _font, _relx, _rely, _relwidth, _relheight, _anchor):
    text = tk.Label(master = placeholder, text=_text, font=40)
    text.place(relx=_relx, rely= _rely, relwidth=_relwidth, relheight=_relheight, anchor = _anchor)

def make_entry(placeholder, _font, _relx, _rely, _relwidth, _relheight, _anchor):
    entry = tk.Entry(placeholder, font = _font)
    entry.place(relx=_relx, rely= _rely, relwidth=_relwidth, relheight=_relheight, anchor = _anchor)

    return entry

def make_label(placeholder, _text, _font, _relx, _rely, _relwidth, _relheight, _anchor):
    label = tk.Label(placeholder, text = _text, font = _font)
    label.place(relx=_relx, rely= _rely, relwidth=_relwidth, relheight=_relheight, anchor = _anchor)


# Windows
def data_analysis_window():
    window = tk.Toplevel(root)
    window.title("Data Analysis")

    c = tk.Canvas(window, height=HEIGHT, width=WIDTH)
    c.pack()

    # Countplot for target
    # Countplot for mean radius
    # Heatmap for cancer dataset
    # Heatmap for correlation of features
    # Heatmap for correlation with target

    offset = 0.15
    b_countplot_target = make_button(c, "Count plot for target", 40, target_count_plot, 0.5, 0.2 - offset, 0.7, 0.1, 'n')
    b_countplot_meanRadius = make_button(c, "Count plot for mean radius", 40, mean_radius_count_plot, 0.5, 0.4 - offset, 0.7, 0.1, 'n')
    b_dataset_heatmap = make_button(c, "Dataset Heatmap", 40, dataset_heatmap, 0.5, 0.6 - offset, 0.7, 0.1, 'n')
    b_feature_corr_heatmap = make_button(c, "Feature Correlation Heatmap", 40, feature_corr_heatmap, 0.5, 0.8 - offset, 0.7, 0.1, 'n')
    b_target_corr_heatmap = make_button(c, "Target Correlation Heatmap", 40, target_corr_heatmap, 0.5, 1 - offset, 0.7, 0.1, 'n')

def train_model_window():
    window = tk.Toplevel(root)
    window.title("Train Model")

    c = tk.Canvas(window, height=HEIGHT, width=WIDTH)
    c.pack()

    # Text
    t_layer_dims = make_text(c, "Layer Dimensions: ", 40, 0.3, 0.1, 0.3, 0.1, 'e')
    t_learning_rate = make_text(c, "Learning Rate: ", 40, 0.28, 0.3, 0.3, 0.1, 'e')
    t_num_iterations = make_text(c, "Number of iterations: ", 40, 0.41, 0.5, 0.5, 0.1, 'e')

    # Entries
    e_layer_dms = make_entry(c, 40, 0.6, 0.1, 0.3, 0.1, 'e')
    e_learning_rate = make_entry(c, 40, 0.6, 0.3, 0.3, 0.1, 'e')
    e_num_iterations = make_entry(c, 40, 0.6, 0.5, 0.3, 0.1, 'e')

    # Button
    b_train_model = make_button(c, "Train Model", 40, lambda: get_train_model_entries(e_layer_dms.get(), e_learning_rate.get(), e_num_iterations.get()), 0.5, 0.7, 0.5, 0.1, 'n')
    b_show_iterations_graph = make_button(c, "Cost Iteration graph", 40, plot_cost_iteration, 0.5, 0.85, 0.3, 0.1, 'n')

def evaluate_window():
    window = tk.Toplevel(root)
    window.title("Evaluate Model")

    c = tk.Canvas(window, height=HEIGHT, width=WIDTH)
    c.pack()

    # Buttons
    make_button(c, "Validation dataset", 50, lambda: evaluate_val(c), 0.5, 0.2, 0.3, 0.1, 'n')
    make_button(c, "Test dataset", 50, lambda: evaluate_test(c), 0.5, 0.6, 0.3, 0.1, 'n')

def predict_window():

    window = tk.Toplevel(root)
    window.title("Predict Model")

    c = tk.Canvas(window, height=HEIGHT, width=WIDTH)
    c.pack()

    # Text
    make_text(c, "Import feature using a file", 40, 0.5, 0.2, 0.6, 0.1, 'n')
    # File Browser
    make_file_browser(c, 0.5, 0.3, 'n', 40)

# --------------------------------------------------------------------------------------------------------------------------------

# Root
root = tk.Tk()
root.title("Cancer Classifier")

# Give the window a size
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

# Buttons
offset = 0.1
b_load_dataset = make_button(canvas, "Load Dataset", 40, load_dataset, 0.5, 0.16 - offset, 0.3, 0.1, 'n')
b_analysis_window = make_button(canvas, "Show Analysis", 40, data_analysis_window, 0.5, 0.32 - offset, 0.3, 0.1, 'n')
b_data_preprocessing = make_button(canvas, "Data Preprocessing", 40, dataset_preprocessing, 0.5, 0.48 - offset, 0.5, 0.1, 'n')
b_train_model = make_button(canvas, "Train Model", 40, train_model_window, 0.5, 0.66 - offset, 0.5, 0.1, 'n')
# b_train_model = make_button(canvas, "Train Model", 40, train_model_window, 0.5, 0.65, 0.5, 0.1, 'n')
b_evaluate_model = make_button(canvas, "Evaluate trained model", 40, evaluate_window, 0.5, 0.82 - offset, 0.5, 0.1, 'n')
b_predict = make_button(canvas, "Predict", 40, predict_window, 0.5, 0.96 - offset, 0.3, 0.1, 'n')

root.mainloop()