import json

# Load the notebook
with open('notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Find the cell with the residuals plot
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'plt.scatter(filtered_data[\'Baseline\'],' in source and 'filtered_data[\'Amplitude\'] - results_whole.detach().numpy()' in source:
            # Fix the residuals calculation
            new_source = source.replace(
                "(filtered_data['Amplitude'] - results_whole.detach().numpy())",
                "(filtered_data['Amplitude'] - results_whole.detach().numpy().flatten())"
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            break

# Save the notebook
with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)