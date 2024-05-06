# Import Modules
import numpy as np
from tqdm import tqdm
import random
import sqlite3


# DATASET
############################################################################################################

# Method to load train set
def load_data(db_path):
    
    # connect to database .db file
    conn = sqlite3.connect(db_path)
    print("Loaded database")

    # fetch data from example table in database
    c = conn.cursor()
    print("Fetching data ...")
    c.execute('SELECT UserID, ItemID, Rating FROM example_table')
    data = c.fetchall()
    conn.close()

    # define user & item indices and ratings values for sparse representation
    user_indices = []
    item_indices = []
    ratings_values = []

    # max user id and item id
    max_user_id = 0
    max_item_id = 0

    # iterate through data and append to user_indices, item_indices and ratings_values
    for user_id, item_id, rating in data:
        user_indices.append(user_id)
        item_indices.append(item_id)
        ratings_values.append(rating)
        max_user_id = max(max_user_id, user_id)
        max_item_id = max(max_item_id, item_id)

    # convert to numpy arrays with correct data types
    user_indices = np.array(user_indices, dtype=np.int32)
    item_indices = np.array(item_indices, dtype=np.int32)
    ratings_values = np.array(ratings_values, dtype=np.float32)

    # print for debugging
    print("Max user id:", max_user_id)
    print("Max item id:", max_item_id)

    return user_indices, item_indices, ratings_values, max_user_id, max_item_id

# normalise indicies to start from 0 and be continuous to N-1
def normalize_indices(indices):
    # get unique indices
    unique_ids = np.unique(indices)
    
    # create mapping from original index to normalised index
    id_to_norm = {id_: i for i, id_ in enumerate(unique_ids)}
    
    # create mapping from normalised index to original index
    norm_to_id = {i: id_ for i, id_ in enumerate(unique_ids)}
    
    # normalise indices
    normalized_indices = np.vectorize(id_to_norm.get)(indices)
    
    # number of unique indices
    num_unique = len(unique_ids)  # Total number of unique indices
    
    return normalized_indices, num_unique, id_to_norm, norm_to_id

# split into train and validation sets
def split_data(user_indices, item_indices, ratings, split_ratio=0.9):
    # set random seed for reproducibility
    np.random.seed(42)
    
    # shuffle data
    indices = np.random.permutation(len(ratings))
    
    # split data
    split_point = int(len(ratings) * split_ratio)
    train_idx, val_idx = indices[:split_point], indices[split_point:]
    
    # assign train and validation data
    train_data = (user_indices[train_idx], item_indices[train_idx], ratings[train_idx])
    val_data = (user_indices[val_idx], item_indices[val_idx], ratings[val_idx])
    
    return train_data, val_data

# import test set
def load_test(db_path):
    
    # connect to database .db file
    conn = sqlite3.connect(db_path)
    print("Loaded database")

    # fetch data from example table in database
    c = conn.cursor()
    print("Fetching data ...")
    c.execute('SELECT UserID, ItemID, TimeStamp FROM example_table')
    data = c.fetchall()
    conn.close()

    # assing user and item indices and timestamps
    test_user_indices, test_item_indices, timestamps = zip(*data)

    return np.array(test_user_indices, dtype=np.int32), np.array(test_item_indices, dtype=np.int32), np.array(timestamps, dtype=np.int32)

# vectorize indices using mapping from normalised indices
def vectorize_indices(indices, mapping):
    
    # report missing indices
    missing_ids = [id for id in indices if id not in mapping]
    if missing_ids:
        print(f"Missing IDs: {set(missing_ids)}")
    
    # vectorize indices using -1 for missing indices
    vectorized_indices = np.array([mapping[id] if id in mapping else -1 for id in indices])

    return vectorized_indices, set(missing_ids)

# save predictions to csv
def save_predictions_to_csv(user_ids, item_ids, predictions, timestamps, filename):
    # Ensure all parts are numpy arrays (in case they are not)
    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    predictions = np.array(predictions)
    timestamps = np.array(timestamps)
    
    # Stack the arrays horizontally
    data_to_save = np.column_stack((user_ids, item_ids, predictions, timestamps))
        
    # Save to CSV
    np.savetxt(filename, data_to_save, delimiter=',', fmt='%d,%d,%.1f,%d')

# MAE AND PREDICTION FUNCTIONS
############################################################################################################

# Method to calculate the mean absolute error from actual and predicted ratings
def calculate_mae(actual, predicted):
    
    # calculate the absolute error between actual and predicted ratings
    abs_err = np.abs(actual - predicted)
    
    # calculate the mean of these absolute errors
    mae = np.mean(abs_err)
    
    return mae

# round prediction to nearest 0.5 in range [0.5, 5]
def round_predictions(predictions):
    rounded_predictions = np.round(predictions * 2) / 2
    return np.clip(rounded_predictions, 0.5, 5.0)

# predict using dot product of user and item features
def predict(user_features, item_features, user_indices, item_indices):
    predictions = np.array([np.dot(user_features[u], item_features[i]) for u, i in zip(user_indices, item_indices)])
    return predictions

# realign predictions to original indices after predicting on nrormalised indices
def realign_predictions(original_user_indices, original_item_indices, normalized_user_indices, normalized_item_indices, predictions, default_value=3.0):
    
    # Initialize the aligned predictions array with the default value
    aligned_predictions = np.full(original_user_indices.shape, default_value, dtype=float)

    # Create a mapping from normalized indices back to original positions
    norm_to_orig_map = {}
    for idx, (norm_u, norm_i) in enumerate(zip(normalized_user_indices, normalized_item_indices)):
        if (norm_u, norm_i) not in norm_to_orig_map:  # avoid overriding if multiple original indices map to the same normalized index
            norm_to_orig_map[(norm_u, norm_i)] = idx

    # Use this map to assign predictions to their corresponding original indices
    for idx in range(len(predictions)):
        norm_u = normalized_user_indices[idx]
        norm_i = normalized_item_indices[idx]
        if (norm_u, norm_i) in norm_to_orig_map:
            orig_idx = norm_to_orig_map[(norm_u, norm_i)]
            aligned_predictions[orig_idx] = predictions[idx]

    return aligned_predictions


# SGD 
############################################################################################################

# method to perform stochastic gradient descent algorithm
def sgd(user_indices, item_indices, ratings, num_users, num_items, num_factors, alpha, beta, iterations):
    # Initialize feature matrices with random seed
    np.random.seed(42)
    user_features = np.random.normal(0, 0.1, (num_users, num_factors))
    item_features = np.random.normal(0, 0.1, (num_items, num_factors))

    # SGD updates 
    for iteration in range(iterations): # number of iterations 
        # tqdm progress bar
        for u, i, r in tqdm(zip(user_indices, item_indices, ratings), desc=f'SGD {iteration+1}/{iterations}', total=len(ratings)):
            
            # predict using dot product, calculate error from rating
            prediction = np.dot(user_features[u], item_features[i])
            error = r - prediction

            # Update rules for features using -2 * error * feature + beta * feature 
            user_features_grad = -2 * error * item_features[i] + beta * user_features[u]
            item_features_grad = -2 * error * user_features[u] + beta * item_features[i]

            # Update features
            user_features[u] -= alpha * user_features_grad
            item_features[i] -= alpha * item_features_grad

    return user_features, item_features


# MAIN
############################################################################################################

if __name__ == '__main__':

    # path to 20M dataset
    path_20M = '../../data/dataset2/train_20M.db'
    
    # original user and item indices, ratings, max user and item ids
    u, i, global_ratings, global_max_user_id, global_max_item_id = load_data(path_20M)

    # Normalize user and item indices, returns normalised indices, number of unique indices, mapping from original to normalised indices and mapping from normalised to original indices
    global_user_indices, global_num_users, user_to_norm, norm_to_user = normalize_indices(u)
    global_item_indices, global_num_items, item_to_norm, norm_to_item = normalize_indices(i)

        
    print("Number of users:", global_num_users)
    print("Number of items:", global_num_items)
    
    # split into train and validation sets with 0.9 split ratio to match test set size ratio
    train_data, val_data = split_data(global_user_indices, global_item_indices, global_ratings, split_ratio=0.9)
    
    # use split data with 1.0 split ratio to create all data array in correct form
    all_data, _ = split_data(global_user_indices, global_item_indices, global_ratings, split_ratio=1)
    
    # print size for debugging
    print("Train data size:",train_data[0].size)
    print("Validation data size:",val_data[0].size)
    print("All data size:",all_data[0].size)

    # path to 20M test dataset
    test_dir_20M = '../../data/dataset2/test_20M.db'

    # Load the dataset 
    test_user_indices, test_item_indices, timestamps = load_test(test_dir_20M)
    test_data = (test_user_indices, test_item_indices, timestamps)
    
    # Normalize test user and item indices, returning missing items and users
    test_user_indices_normalized, missing_users = vectorize_indices(test_user_indices, user_to_norm)
    test_item_indices_normalized, missing_items = vectorize_indices(test_item_indices, item_to_norm)
    
    # print relevant information for debugging
    print("Test data size: ", len(test_data[0]))
    print("All data size: ", len(all_data[0]))
    print("Ratio All Data / Test:", len(test_data[0]) / len(all_data[0]))
    print("Ratio Train / Val:", len(val_data[0]) / len(train_data[0]))

    print("Num users:", len(np.unique(test_user_indices_normalized)))
    print("Num items:", len(np.unique(test_item_indices_normalized)))
    
    # Optimised Parameters for SGD based on validation set
    num_factors = 60  # Latent factors
    alpha = 0.0075      # Learning rate
    beta = 0.03      # Regularization
    iterations = 2   # Number of iterations

    # perform SGD on all data, returning user and item factors
    sgd_user_factors, sgd_item_factors = sgd(all_data[0], all_data[1], all_data[2], global_num_users, global_num_items, num_factors, alpha, beta, iterations)

    # default value for missing data
    default_prediction_value = 3 

    # predict on normalised test data
    predictions = predict(sgd_user_factors, sgd_item_factors, test_user_indices_normalized, test_item_indices_normalized)

    # Replace predictions for missing indices with the default value
    predictions[test_item_indices_normalized == -1] = default_prediction_value
    
    # realign predictions to original indices
    aligned_predictions = realign_predictions(test_user_indices, test_item_indices, test_user_indices_normalized, test_item_indices_normalized, predictions)
    
    # round predictions to nearest 0.5 in range [0.5, 5]
    final_predictions = round_predictions(aligned_predictions)
    
    # save predictions to csv 
    path = 'results.csv'
    save_predictions_to_csv(test_user_indices, test_item_indices, aligned_predictions, timestamps, path)