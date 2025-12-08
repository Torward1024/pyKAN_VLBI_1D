import json

# Load the notebook
with open('notebook.ipynb', 'r') as f:
    notebook = json.load(f)

# Function to replace in source
def replace_in_source(cell, old, new):
    if 'source' in cell:
        cell['source'] = [line.replace(old, new) for line in cell['source']]

# Replacements
replace_in_source(notebook, "y_data = rest_data[['Re', 'Im']].values", "y_data = rest_data[['Re', 'Im', 'Amplitude', 'Phase']].values")
replace_in_source(notebook, "width=[3, 12, 12, 2]", "width=[3, 12, 12, 4]")
replace_in_source(notebook, "results_amplitude = np.sqrt(results_denormalized[:, 0]**2 + results_denormalized[:, 1]**2)", "results_amplitude = results_denormalized[:, 2]")
replace_in_source(notebook, "results_phase = np.arctan2(results_denormalized[:, 1], results_denormalized[:, 0])", "results_phase = results_denormalized[:, 3]")
replace_in_source(notebook, "results_whole = np.sqrt(results_whole_denormalized[:, 0]**2 + results_whole_denormalized[:, 1]**2)", "results_whole = results_whole_denormalized[:, 2]")
replace_in_source(notebook, "results_phase_whole = np.arctan2(results_whole_denormalized[:, 1], results_whole_denormalized[:, 0])", "results_phase_whole = results_whole_denormalized[:, 3]")

# Save the notebook
with open('notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)