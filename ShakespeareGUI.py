import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, OptionMenu, StringVar, W, ttk
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.image as mpimg
import time

batch_factor = 16


# Set base directory
base_dir = Path(__file__).resolve().parent

# Paths relative to the base directory
path_imposters = base_dir / 'metadata' / 'imposters_pairs.json'
path_index = base_dir / 'metadata' / 'training_state.json'
shakespeare_dir = base_dir / 'shakespeare'
output_dir = base_dir / 'outputs'
path_output_file= output_dir / 'shakespeare_evaluation_results.csv'
path_model_graph= output_dir / 'model_graph.jpeg'


root = tk.Tk()
root.title("Outlier Detection in Shakespeare Works")
root.geometry('330x200')  # Set a default size for the root window
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=6)
style.configure('TLabel', font=('Helvetica', 12), padding=6)
frame_dropdowns = ttk.Frame(root)
frame_dropdowns.pack(pady=10, fill="x", padx=10)  # This frame contains the dropdown menus

# Button frame to hold buttons side by side
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)  # Add some vertical padding


# Load the imposter pairs from a JSON file
def load_imposters(file_path):
    with open(file_path, 'r') as file:
        imposter_pairs = json.load(file)
    return imposter_pairs


# Load the current pair index from a JSON file
def load_current_index(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        current_index = data['current_pair_index']
    return current_index


# Define a function to clean text
def clean_text(text):
    if not tf.is_tensor(text):
        text = tf.constant(text, dtype=tf.string)
    # Normalize text by converting to lowercase
    text = tf.strings.lower(text)
    # Remove '\n', '\r', and '\t'
    text = tf.strings.regex_replace(text, r"[\n\r\t]", " ")
    # Remove excessive whitespace
    text = tf.strings.regex_replace(text, r"\s+", " ")
    # Optionally, remove numbers if they're deemed irrelevant. Uncomment if needed.
    text = tf.strings.regex_replace(text, r"\d+", " ")
    # Remove special characters except apostrophes and periods (which might be relevant for sentence boundary detection)
    text = tf.strings.regex_replace(text, r"[^a-z0-9'. ]", " ")
    # Trim leading and trailing whitespace
    text = tf.strings.strip(text)
    return text.numpy().decode('utf-8')  # Convert from tensor to string


def load_texts_from_directory(directory_path, aggregate_by_subfolder=True):
    """
    Load texts from files in the given directory, optionally aggregating them by subfolder or by filenames if no subfolders exist.

    Parameters:
    - directory_path (str): Path to the directory containing text files or subdirectories with text files.
    - aggregate_by_subfolder (bool): If True, returns a dictionary where each key is a subfolder name (author),
      and its value is a list of text contents from that author. If no subfolders are present, or if False,
      returns a dictionary with filenames as keys and text contents as values.

    Returns:
    - texts_dictionary (dict): Depending on `aggregate_by_subfolder` and the presence of subfolders, returns a dictionary
      with either subfolder names or filenames as keys and lists of text contents as values.
    """
    texts_dictionary = {}
    directory_path = Path(directory_path)  # Ensure directory_path is a Path object
    subfolders_exist = any((directory_path / i).is_dir() for i in directory_path.iterdir())

    # Determine whether to aggregate by subfolder or directly by filenames
    aggregate_by = aggregate_by_subfolder and subfolders_exist

    for root, dirs, files in os.walk(str(directory_path)):
        root_path = Path(root)  # Convert the root back to a Path object
        if not aggregate_by and root_path != directory_path:
            break
        for name in files:
            file_path = root_path / name
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with file_path.open('r', encoding='iso-8859-1') as f:
                    text = f.read()
            key = root_path.name if aggregate_by else name
            if key not in texts_dictionary:
                texts_dictionary[key] = []
            texts_dictionary[key].append(text)
    return texts_dictionary


def detect_outliers_with_isolation_forest(distance_matrix, filenames, num_trees=100, outlier_threshold=0.2,
                                          percentile_threshold=90):
    # Calculate the sum of distances for each file (assuming each row corresponds to a file)
    distance_sums = np.sum(distance_matrix, axis=1)
    high_distance_indices = np.where(distance_sums > np.percentile(distance_sums, percentile_threshold))[0]

    # Isolation Forest for additional outlier detection
    model = IsolationForest(n_estimators=num_trees, contamination=outlier_threshold)
    model.fit(distance_matrix)
    is_outlier = model.predict(distance_matrix)
    # Combine custom high distance detection with Isolation Forest results
    combined_outliers = np.zeros(len(filenames), dtype=bool)
    combined_outliers[is_outlier] = True  # Mark Isolation Forest detected outliers
    combined_outliers[high_distance_indices] = True  # Mark high distance files as outliers

    # Prepare scores just for visualization purposes (can be based on Isolation Forest decision function or distance sums)
    scores = model.decision_function(distance_matrix)  # Negative scores to make higher values more 'outlying'

    return filenames, scores, combined_outliers, distance_sums








def visualize_distances_clusters(distance_matrix, filenames, is_outlier, scores, window, title):
    # Perform clustering on the data
    kmeans = KMeans(n_clusters=2, random_state=42).fit(scores.reshape(-1, 1))

    # Visualize the clusters
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(scores, distance_matrix, c=kmeans.labels_, cmap='viridis')

    # Annotate text names above the points
    for i, txt in enumerate(filenames):
        if is_outlier[i]:
            ax.annotate(txt, (scores[i], distance_matrix[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        else:
            ax.annotate(str(scores[i]), (distance_matrix[i], 0), textcoords="offset points", xytext=(0, 10), ha='center')

    ax.set_title(title)  # Set title
    ax.set_xlabel('Outlier Score (Smaller is more anomalous)')  # Set x-label
    ax.set_ylabel('Distance')  # Set y-label
    ax.legend(*scatter.legend_elements(), title='Cluster')  # Add legend for clusters
    ax.grid(True)

    # Show plot
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=1)

def normalize_data(data):
    return normalize(np.array(data).reshape(-1, 1), axis=0)


def perform_clustering(data_normalized):
    silhouette_scores = []
    cl_all = []
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data_normalized.reshape(-1, 1))
    model = KMedoids(n_clusters=2, random_state=42).fit(data_normalized)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(data_normalized, labels)
    silhouette_scores.append(silhouette_avg)
    cl_all.append(labels)
    cl_all = np.array(cl_all).T  # Convert to 2D array

    return cl_all, silhouette_scores




def visualize_clusters_new(cl_all, filenames, sorted_results, window, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.imshow(cl_all, aspect='auto', interpolation='none', cmap='viridis')
    plt.colorbar(scatter)

    for i, txt in enumerate(filenames):
        ax.annotate(txt, (0, i), textcoords="offset points", xytext=(0, 10), ha='center')

    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('File Name')
    ax.set_yticks(range(len(filenames)))
    ax.grid(False)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=1)


shakespeare_texts_dictionary = load_texts_from_directory(shakespeare_dir, aggregate_by_subfolder=False)
shakespeare_texts_list = [text for texts in shakespeare_texts_dictionary.values() for text in texts]
shakespare_filenames = list(shakespeare_texts_dictionary.keys())
shakespeare_preprocessed_texts = [clean_text(text) for text in shakespeare_texts_list]


def on_button_clicked(*args):
    time.sleep(4)

    results_window = tk.Toplevel(root)
    results_window.title("Model Results")

    try:
        # Identify the iteration of the selected pair
        selected_pair = [imposter1_var.get(), imposter2_var.get()]
        if selected_pair in valid_imposters:
            iteration_number = valid_imposters.index(selected_pair)
        else:
            selected_pair_reversed = [imposter2_var.get(), imposter1_var.get()]
            if selected_pair_reversed in valid_imposters:
                iteration_number = valid_imposters.index(selected_pair_reversed)
            else:
                iteration_number = "Pair not found within the valid range."
                messagebox.showerror("Error", "Pair not found within the valid range.")
                return

        distance_results_path = output_dir / f"dist_mat_{iteration_number}.npy"
        distance_results = np.load(distance_results_path, allow_pickle=True)

        window_title = f"{imposter1_var.get()} VS {imposter2_var.get()}"
        filenames, scores, is_outlier, distance_sums = detect_outliers_with_isolation_forest(distance_matrix=distance_results, filenames=shakespare_filenames , num_trees=100, outlier_threshold=0.2,percentile_threshold=90)
        visualize_distances_clusters(distance_sums, filenames, is_outlier,scores, window=results_window, title=window_title)


    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        results_window.destroy()

# Function to clear results
def clear_results():
    # This will destroy any open results windows
    for widget in root.winfo_children():
        if isinstance(widget, tk.Toplevel):
            widget.destroy()
# Load data
imposters = load_imposters(path_imposters)
current_index = load_current_index(path_index)

# Filter imposters up to the current index
valid_imposters = imposters[:current_index + 1]

# Extract all unique writers from pairs up to the current index
all_writers = set()
for pair in valid_imposters:
    all_writers.update(pair)
all_writers = sorted(list(all_writers))

# Dropdowns for selecting imposter pairs
imposter1_var = StringVar(root)
imposter1_var.set('Select Imposter 1')
imposter2_var = StringVar(root)
imposter2_var.set('Select Imposter 2')

# Dropdown for selecting imposter 1
label_imposter1 = Label(frame_dropdowns, text="Imposter 1:")
label_imposter1.pack(fill="x")  # Filling horizontally
menu_imposter1 = OptionMenu(frame_dropdowns, imposter1_var, *all_writers)
menu_imposter1.pack(fill="x")  # Filling horizontally

# Dropdown for selecting imposter 2
label_imposter2 = Label(frame_dropdowns, text="Imposter 2:")
label_imposter2.pack(fill="x")  # Filling horizontally
menu_imposter2 = OptionMenu(frame_dropdowns, imposter2_var, "")  # Initialize with all writers initially
menu_imposter2.pack(fill="x")  # Filling horizontally

def analyzer_overall_results():
    df = pd.read_csv(path_output_file)
    data = df.select_dtypes(include=[np.number])
    filenames = df.iloc[:, 0]

    # Filter out columns with zero standard deviation
    std_devs = np.std(data, axis=0)
    data = data.loc[:, std_devs > 0]

    # Normalize the data
    data_normalized = normalize(data, axis=0)

    # Initialize variables for storing results
    silhouette_scores = []
    cl_all = np.zeros((data_normalized.shape[0], data_normalized.shape[1] - 5))

    # Perform clustering on incrementally more features
    for nn in range(5, data_normalized.shape[1]):
        current_data = data_normalized[:, :nn]
        current_data = normalize(current_data, axis=0)
        model = KMedoids(n_clusters=2, random_state=42).fit(current_data)
        cl_all[:, nn - 5] = model.labels_
        silhouette_avg = silhouette_score(current_data, model.labels_)
        silhouette_scores.append(silhouette_avg)

    # Extract the final clusters and create a DataFrame to store results
    final_clusters = cl_all[:, -1]
    results_df = pd.DataFrame({
        'Filename': filenames,
        'Cluster': final_clusters
    })

    # Calculate mean cluster labels
    mean_cluster_labels = cl_all.mean(axis=1)

    # Sort filenames based on mean cluster labels
    sorted_indices = np.argsort(mean_cluster_labels)
    sorted_filenames = filenames[sorted_indices]
    sorted_mean_labels = mean_cluster_labels[sorted_indices]

    # Create a DataFrame to display sorted results
    sorted_df = pd.DataFrame({
        'Filename': sorted_filenames,
        'MeanClusterLabel': sorted_mean_labels
    })

    # Plot the cluster assignments over iterations
    plt.figure(figsize=(12, 10))
    plt.imshow(cl_all[sorted_indices, :], aspect='auto', interpolation='none', cmap='viridis')
    plt.colorbar()
    plt.title('Cluster Assignments Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('File Name')
    plt.title('Overall Model Results')
    plt.yticks(range(len(filenames)), sorted_df['Filename'])  # Use sorted filenames for y-axis
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def display_overall_results():
    time.sleep(15)

    img = mpimg.imread(path_model_graph)
    plt.imshow(img)
    plt.axis('off')  # Turn off axis labels
    plt.title('Clustered Data')
    plt.show()
def update_dropdown2(*args):
    selected_imposter1 = imposter1_var.get()
    # Filter pairs up to the current index
    filtered_pairs = [pair for pair in imposters[:current_index+1] if selected_imposter1 in pair]
    # Find all partners of the selected writer
    partners = [pair[0] if pair[1] == selected_imposter1 else pair[1] for pair in filtered_pairs]

    # Update the second dropdown
    menu_imposter2['menu'].delete(0, 'end')
    if not partners:
        menu_imposter2['menu'].add_command(label="No available partners", command=lambda: imposter2_var.set(""))
    else:
        for partner in partners:
            menu_imposter2['menu'].add_command(label=partner, command=lambda partner=partner: imposter2_var.set(partner))


imposter1_var.trace('w', update_dropdown2)

button_run = Button(button_frame, text="Display Imposters Model Results", command=on_button_clicked)
button_run.pack(side='left', padx=5)  # Pack buttons side by side with some padding
# Button to clear results
button_clear = Button(button_frame, text="Overall Model Results", command=display_overall_results)
button_clear.pack(side='left', padx=5)
# Start the Tkinter event loop
root.mainloop()
