from os.path import join

import pandas as pd

model_feat_nico = pd.read_csv(join("data", "Model_features08-12-18_nico.csv"))
sub_data_nico = pd.read_csv(join("data", "sub_dataset_energy07-57-54_nico.csv"))

model_feat_gui = pd.read_csv(join("data", "Model_features08-15-31_guillaume.csv"))
sub_data_gui = pd.read_csv(join("data", "sub_dataset_energy07-40-32_guillaume.csv"))


def extract_energy_consumed(sub_data:pd.DataFrame)->list:
    """Extract the column of a sub data set produced with a codecarbon framework"""
    energy_consumed = list(sub_data["energy_consumed"])
    energy_consumed.reverse()
    for i in range(0, len(energy_consumed)-1):
        energy_consumed[i] -=  energy_consumed[i+1] # Normalise the energy as it is cumulative
    energy_consumed.reverse()
    return energy_consumed

def add_energy_col(data_set:pd.DataFrame, energy:list)-> pd.DataFrame:
    """add the consumed energy to a dataframe of predictors"""
    data_set['energy_consumed'] = energy
    return data_set

energy_n = extract_energy_consumed(sub_data_nico)
sub_data_e_n = add_energy_col(model_feat_nico, energy_n)


energy_g = extract_energy_consumed(sub_data_gui)
sub_data_e_g = add_energy_col(model_feat_gui, energy_g)

def concat_subsets(subsets:list)->pd.DataFrame:
    """concatenate two subsets together"""
    return pd.concat(subsets, axis=0).reset_index(drop=True)

full_data_set = concat_subsets(sub_data_e_n, sub_data_e_n)

def drop_memory_features(full_data_set:pd.DataFrame)->pd.DataFrame:
    """/!\ for now memory features are not dynamics, it is in the TODO, therefore drop these columns"""
    return full_data_set.drop(columns = ["memory_available_B", "swap_free_B"])

def full_dataset_pipeline(pair_datas:list)->pd.DataFrame:
    """Gets the entire enery of an ML model dataset.

    Parameters
    ----------
    pair_datas : list[Tuple[pd.DataFrame]]
        a list of tuples of pd.DataFrame, with index 0 containing a `consumed_energy` column, index 1 are the extracted features

    Returns
    -------
    pd.DataFrame
        The fully concatenated dataset.
    """
    energy_cols = [extract_energy_consumed(df_e) for df_e, _ in pair_datas]
    energy_dfs = [add_energy_col(pair_datas[1][i], energy_cols[i]) for i in range(len(energy_cols))]
    full_data_set = concat_subsets(energy_dfs)
    return drop_memory_features(full_data_set)
        
def rename_categorical_cols(data_to_rename:pd.DataFrame)->pd.DataFrame:
    """Rename all categorical predictors with a suffix `name_`"""
    return data_to_rename.rename(columns = {'CPU_vendor_id':'name_CPU_vendor_id', 
                                            'core_architecture':'name_core_architecture',
                                            'os':'name_os',
                                            'model_name' : 'name_model'
                                            })
    
    
    
    

