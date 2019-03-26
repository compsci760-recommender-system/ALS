import random
import pandas as pd
import numpy as np

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

from implicit_als import implicit_als

# -------------------------
# THE CODE IN THIS FILE SHOULD BE CHANGED WITH DATA FOR PREPROSESSING OUR OWN DATASET
# -------------------------

raw_data = pd.read_csv('data/usersha1-artmbid-artname-plays.tsv', sep='\t')
raw_data = raw_data.drop(raw_data.columns[1], axis=1)
raw_data.columns = ['user', 'artist', 'plays']

# Drop rows with missing values
data = raw_data.dropna()

# Convert artists names into numerical IDs
data['user_id'] = data['user'].astype("category").cat.codes
data['artist_id'] = data['artist'].astype("category").cat.codes

# Create a lookup frame so we can get the artist names back in
# readable form later.
item_lookup = data[['artist_id', 'artist']].drop_duplicates()
item_lookup['artist_id'] = item_lookup.artist_id.astype(str)

data = data.drop(['user', 'artist'], axis=1)

# Drop any rows that have 0 plays
data = data.loc[data.plays != 0]

# Create lists of all users, artists and plays
users = list(np.sort(data.user_id.unique()))
artists = list(np.sort(data.artist_id.unique()))
plays = list(data.plays)

# Get the rows and columns for our new matrix
rows = data.user_id.astype(int)
cols = data.artist_id.astype(int)

# Contruct a sparse matrix for our users and items containing number of plays
data_sparse = sparse.csr_matrix((plays, (rows, cols)), shape=(len(users), len(artists)))

# Run the algorithm on the data
implicit_als(data_sparse)
