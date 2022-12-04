import pandas as pd
from sklearn.preprocessing import OneHotEncoder


#Data preprocessing
dataset = pd.read_csv('energy_dataset.csv')
dataset.filter(regex='name')
cat_cols = [col for col in dataset.columns if 'name' in col]

#One Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore') #### peut-etre mettre error pour handle_unknown

encoder_df = pd.DataFrame(encoder.fit_transform(df[['team']]).toarray())

#merge one-hot encoded columns back with original DataFrame
final_df = df.join(encoder_df)





#Linear Regression Method




#XGBoost