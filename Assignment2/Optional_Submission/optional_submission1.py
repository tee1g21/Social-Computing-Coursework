# import modules
import numpy as np
from tqdm import tqdm
import random
import sqlite3

# HELPER METHODS
############################################################################################################

# MAE and predict methods
def calculate_mae(actual, predicted):
    """
    Parameters:
    - actual_ratings: np.array, the actual ratings.
    - predicted_ratings: np.array, the predicted ratings.
    """
    # calculate the absolute error between actual and predicted ratings
    abs_err = np.abs(actual - predicted)
    
    # calculate the mean of these absolute errors
    mae = np.mean(abs_err)
    
    return mae

# round prediction to nearest 0.5 in range [0.5, 5]
def round_predictions(predictions):
    rounded_predictions = np.round(predictions * 2) / 2
    return np.clip(rounded_predictions, 0.5, 5.0)

# predict rating for a user-item pair
def predict_rating(user_id, item_id, user_factors, item_factors):
    user_vector = user_factors[user_id]
    item_vector = item_factors[item_id]
    return np.dot(user_vector, item_vector)

# generate predictions for all user-item pairs in the validation set
def generate_predictions(validation_dict, user_factors, item_factors):
    predictions = []
    actual_ratings = []
    for (user_id, item_id), actual_rating in validation_dict.items():
        predicted_rating = predict_rating(user_id, item_id, user_factors, item_factors)
        if predicted_rating is not None:
            predictions.append(predicted_rating)
            actual_ratings.append(actual_rating)
        else:
            predictions.append(np.nan)  # Appending NaN for undefined predictions
            actual_ratings.append(actual_rating)
    return np.array(predictions), np.array(actual_ratings)

def generate_testset_predicitons(test_data, user_factors, item_factors):
    # Initialize an empty list to hold predictions
    predictions = []
    
    # Assume test_data is a numpy array with columns: [userID, itemID, timestamp]
    for user, item, _ in test_data:
        user = int(user)
        item = int(item)
        prediction = predict_rating(user, item, user_factors, item_factors)
        predictions.append(prediction)
    
    return np.array(predictions)

# get num users and items
def get_max_users_items(data):
    max_user = max(user_id for user_id, item_id in data.keys()) + 1
    max_item = max(item_id for user_id, item_id in data.keys()) + 1
    return max_user, max_item


# SGD
############################################################################################################

def init_factors(num_factors, max_id):
    # Initialize factor vectors with small random numbers
    factors = {}
    np.random.seed(42)
    for i in range(1, max_id + 1):  # IDs start at 1
        factors[i] = np.random.normal(scale=0.05, size=num_factors).astype(np.float64)
    return factors

def sgd_update(user_factors, item_factors, user_id, item_id, actual_rating, alpha, beta, num_factors):
    # Predict the rating    
    prediction = np.dot(user_factors[user_id], item_factors[item_id])
    error = actual_rating - prediction
    
    # Update factors
    user_factors[user_id] += alpha * (error * item_factors[item_id] - beta * user_factors[user_id])
    item_factors[item_id] += alpha * (error * user_factors[user_id] - beta * item_factors[item_id])
    return user_factors, item_factors

def matrix_factorization_SGD(train_data, num_factors, alpha, beta, num_epochs):
    # Initialize factors
    
    max_user, max_item = get_max_users_items(train_data)
    print("Number of users:", max_user, ", Number of items:", max_item)
    
    user_factors = init_factors(num_factors, max_user)
    item_factors = init_factors(num_factors, max_item)
    
    # Perform SGD
    for epoch in tqdm(range(num_epochs), desc='SGD iterations', total=num_epochs):
        for (user_id, item_id), rating in train_data.items():
            user_factors, item_factors = sgd_update(
                user_factors, item_factors, user_id, item_id, rating, alpha, beta, num_factors)
    
    return user_factors, item_factors


# ALS
############################################################################################################

def init_factors(num_factors, size):
    """ Initialize factors as random normal variables. """
    np.random.seed(42)
    return np.random.normal(scale=0.1, size=(size, num_factors))

def update_factors(fixed_factors, ratings_dict, num_factors, lambda_reg):
    num_entities = fixed_factors.shape[0]
    new_factors = np.zeros_like(fixed_factors)
    
    for i in range(num_entities):
        A = np.zeros((num_factors, num_factors))
        b = np.zeros(num_factors)
        for j, rating in ratings_dict.get(i, {}).items():
            A += np.outer(fixed_factors[j], fixed_factors[j])
            b += rating * fixed_factors[j]
        A += lambda_reg * np.eye(num_factors)
        new_factors[i] = np.linalg.solve(A, b)
    
    return new_factors

def run_als(train_data, num_factors, lambda_reg, iterations):
       
    num_users, num_items = get_max_users_items(train_data)
    
    user_factors = init_factors(num_factors, num_users)
    item_factors = init_factors(num_factors, num_items)
    
    for _ in tqdm(range(iterations), desc='ALS iterations', total=iterations):
        user_ratings = {u: {} for u in range(num_users)}
        item_ratings = {i: {} for i in range(num_items)}
        for (u, i), r in train_data.items():
            user_ratings[u][i] = r
            item_ratings[i][u] = r
        
        user_factors = update_factors(item_factors, user_ratings, num_factors, lambda_reg)
        item_factors = update_factors(user_factors, item_ratings, num_factors, lambda_reg)

    return user_factors, item_factors

if __name__ == '__main__':

    # IMPORT DATASET
    ############################################################################################################

    conn = sqlite3.connect('../../Specification/D1/train_100k.db') #stores the main 100k test
    #conn = sqlite3.connect('../Specification/D2/train_20M.db') #stores the main 100k test
    print("loaded database")

    c = conn.cursor()

    #Fetch data
    print("fetching data ...")
    c.execute('SELECT UserID, ItemID, Rating FROM example_table')
    data = c.fetchall()

    #Close the connection
    conn.close()

    # Extract matrix defining data
    train_data = {}
    max_user_id = 0
    max_item_id = 0

    for user_id, item_id, rating in data:
        if rating > 0:  # Assuming we only care about positive ratings
            train_data[(user_id, item_id)] = rating
            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, item_id)
            
    # import test set

    # 100k dataset
    test_dir_100K = '../../Specification/D1/test_100k_withoutratings.csv'

    # Load the dataset
    def load_data_np(filepath):
        return np.loadtxt(filepath, delimiter=',', skiprows=0, dtype='float32')
    
    # Load the dataset (excluding the header if present)
    test_data = load_data_np(test_dir_100K)


    # TRAIN
    ############################################################################################################

    # Run SGD on all data
    num_factors = 20
    alpha = 0.01
    beta = 0.02
    iterations = 10

    sgd_user_factors, sgd_item_factors = matrix_factorization_SGD(train_data, num_factors, alpha, beta, iterations)
    sgd_test_predictions = round_predictions(generate_testset_predicitons(test_data, sgd_user_factors, sgd_item_factors))

    # Run ALS on all data
    num_factors = 2  
    lambda_reg = 0.5  
    iterations = 10   

    als_user_factors, als_item_factors = run_als(train_data, num_factors, lambda_reg, iterations=10)
    als_test_predictions = round_predictions(generate_testset_predicitons(test_data, als_user_factors, als_item_factors))


    # weighted predictions
    weight_sgd = 0.5  
    weight_als = 0.5 

    # sgd_predictions and als_predictions are arrays of the same shape containing the predicted ratings
    weighted_test_predictions = round_predictions((weight_sgd * sgd_test_predictions) + (weight_als * als_test_predictions))


    # SAVE PREDICTIONS TO CSV
    ############################################################################################################

    # create numpy array with the predicted ratings
    predicted_testset =  complete_data = np.hstack((
        test_data[:, :2],  # UserID and ItemID columns
        weighted_test_predictions.reshape(-1, 1),  # Predicted ratings
        test_data[:, 2:]))  # Timestamp column

    path = 'test.csv'
    np.savetxt(path, predicted_testset, delimiter=",", fmt='%d,%d,%.1f,%d')