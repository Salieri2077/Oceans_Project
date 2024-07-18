import re
import pandas as pd

def fetch_word(file_name, model_name):
    with open(file_name, 'r') as file:
        data = file.read()
    
    # pattern = r'mse:(\d+\.\d+)'
    pattern = r'mae:(\d+\.\d+)'
    matches = re.findall(pattern, data)
    MSE = float(matches[0])  # Convert the first match to float
    return MSE

# Fetch MSE data
models = ['LSTM', 'GRU','VanRNN','transformer']
time_steps = [24,36,48,72,96]

# Create a DataFrame to store MSE data
mse_data = pd.DataFrame(index=time_steps, columns=models)

# Populate the DataFrame with MSE values
for model in models:
    for time_step in time_steps:
        mse_value = fetch_word(f'{model}_{time_step}.txt', f'{model}-{time_step}')
        mse_data.at[time_step, model] = mse_value

# Export DataFrame to Excel
# excel_file = 'mse_data.xlsx'
excel_file = 'mae_data.xlsx'
mse_data.to_excel(excel_file, float_format='%.3f')  # Set the float format to 3 decimal places

print("MSE data has been exported to", excel_file)
