'''
CEBRA for behavior --cluster behavior data
'''
import cebra
# Create a .npz file
import numpy as np
import pandas as pd
import cebra.models

# Transform csv to h5 file
WorkingFolder = r'D:\project\BCI\data\data_30\20230520FOV\Behavior\CEBRA'   # path of behavior_data
BehaviorFile = f'{WorkingFolder}\Full_Inf.csv'

Behavior = pd.read_csv(BehaviorFile, sep=",") # read csv file
Behavior = Behavior.drop("Unnamed: 0", axis=1)

Behavior.to_hdf(f'{WorkingFolder}\Full_Inf.h5', key="Behavior_Inf")

# Read data from folder for CEBRA
Behavior_data = cebra.load_data(file=f'{WorkingFolder}\Full_Inf.h5', key="Behavior_Inf")
conditional = "time" # set CEBRA model as time

#  CEBRA model initialization
#  Parameter setting
model_architecture = "offset10-model"
out_dim = 8
time_offsets = 10
learning_rate = 0.001
max_iterations = 10000
#  CEBRA model
cebra_model = cebra.CEBRA(
    model_architecture = model_architecture,
    temperature_mode="auto",
    learning_rate = learning_rate,
    max_iterations = max_iterations,
    time_offsets = time_offsets,
    output_dimension = out_dim,
    device = "cuda_if_available",
    verbose = False
)
print(cebra_model)
# Training
cebra_model.fit(Behavior_data)

# Save model
cebra_model.save(f'{WorkingFolder}\cebra_model.pt')
''' 
# Grid search
# 1. Define the parameters, either variable or fixed
params_grid = dict(
    output_dimension = [3, 16],
    learning_rate = [0.001],
    time_offsets = 5,
    max_iterations = 5,
    temperature_mode = "auto",
    verbose = False)
# 2. Define the datasets to iterate over
datasets = {"dataset1": Behavior_data}
# 3. Create and fit the grid search to your data
grid_search = cebra.grid_search.GridSearch()
grid_search.fit_models(datasets=datasets, params=params_grid, models_dir="saved_models")
# 4. Get the results
df_results = grid_search.get_df_results(models_dir="saved_models")
# 5. Get the best model for a given dataset
best_model, best_model_name = grid_search.get_best_model(dataset_name="dataset2", models_dir="saved_models")
'''

