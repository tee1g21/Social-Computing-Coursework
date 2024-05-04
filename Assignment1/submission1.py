import numpy as np

def calculate_mae(actual, predicted):
   
    # calculate the absolute error between actual and predicted ratings
    abs_err = np.abs(actual - predicted)
    
    # calculate the mean of these absolute errors
    mae = np.mean(abs_err)
    
    return mae

##cosine similarity
def cosine_similarity(matrix):

    # calculate the dot product of matrix and the transpose
    dot_product = np.dot(matrix, matrix.T)

    # calculate the norm of each vector in matrix
    norm = np.linalg.norm(matrix, axis=1)

    # calculate the outer product of the norms
    norm_outer = np.outer(norm, norm)

    # calculate cosine similarity
    cosine_sim = dot_product / norm_outer

    # Replace any NaN values with 0
    cosine_sim = np.nan_to_num(cosine_sim)

    return cosine_sim

## recommends popular items for new users
def recommend_popular(user_id, item_user_matrix, N=10):

    # calculate sum of ratings for items
    popularity = np.sum(item_user_matrix, axis=1)

    # return indices of the top n popular items
    pop_items = np.argsort(popularity)[-N:]

    # If user has already rated some popular items, remove from list
    rated_items = np.where(user_item_matrix[user_id, :] != 0)[0]
    recommendations = [item for item in pop_items if item not in rated_items]

    return recommendations

def predict_rating_user(user_id, item_id, user_similarity, user_item_matrix, k=28, default_rating=3.0, similarity_threshold=0.025):

    # prevent out of bound indices
    if user_id >= user_similarity.shape[0] or item_id >= user_item_matrix.shape[1]:
        return default_rating
    
    # check if user has already rated items
    if np.count_nonzero(user_item_matrix[user_id, :]) == 0:

        # if new user, recommend a popular item
        recommendations = recommend_popular(user_id, user_item_matrix.T)
        
        if item_id in recommendations:
            # if the item is among the recommended items, return a high rating
            return 5
        else:
            # if not return the default rating
            return default_rating
    
    # calculate user bias
    user_means = np.true_divide(user_item_matrix.sum(1), (user_item_matrix != 0).sum(1))
    user_bias = np.nan_to_num(user_means - np.mean(user_means))  
    
    # apply bias correction to ratings
    similarities = user_similarity[user_id, :]
    ratings = user_item_matrix[:, item_id] - user_bias  
    
    # apply similarity threshold and get k most similar users
    valid_indices = (ratings != -user_bias) & (similarities > similarity_threshold)
    valid_similarities = similarities[valid_indices]
    valid_ratings = ratings[valid_indices]
    
    # get indices of k most similar users
    k_similar_users = np.argsort(-valid_similarities)[:k]
    k_similarities = valid_similarities[k_similar_users]
    k_ratings = valid_ratings[k_similar_users]
    
    # Check if no similar users or if the sum of their similarities is 0
    if k_similarities.size == 0 or np.sum(k_similarities) == 0:
        return default_rating
    
    # Calculate weighted sum of ratings from similar users
    weighted_sum = np.dot(k_similarities, k_ratings)
    sum_of_weights = np.sum(k_similarities)
    
    # compute predicted rating with user bias
    predicted_rating = weighted_sum / sum_of_weights + user_bias[user_id]  

    # round to acceptable range
    return np.clip(round(predicted_rating * 2) / 2, 0.5, 5.0)






## Load the datasets

# dataset locations
train_dir = 'Specification/D1/train_100k_withratings.csv'
test_dir = 'Specification/D1/test_100k_withoutratings.csv'

# userid, itemid, rating, and timestamp
train_data = np.genfromtxt(train_dir, delimiter=',', skip_header=0)
print(train_data.shape)
#userid, itemid, and timestamp 
test_data = np.genfromtxt(test_dir, delimiter=',', skip_header=0)
print(test_data.shape)



## user-item matrix

# array dimensions 
num_users = int(np.max(train_data[:, 0]))  
num_items = int(np.max(train_data[:, 1]))  

# initialise matrices
user_item_matrix = np.zeros((num_users, num_items))
item_user_matrix = np.zeros((num_items, num_users))

# populate user-item matrix
for entry in train_data:
    user_id, item_id, rating, _ = entry
    user_item_matrix[int(user_id)-1, int(item_id)-1] = rating

# generate item-user matrix from transpose
item_user_matrix = user_item_matrix.T

## calculate cosine similarity
user_sim_matrix = cosine_similarity(user_item_matrix)
item_sim_matrix = cosine_similarity(item_user_matrix.T)


# predict data in test set
predictions = []
for user_id, item_id, _ in test_data:
    user_id, item_id = int(user_id)-1, int(item_id)-1  # adjust for 0-based indexing
    pred_rating = predict_rating_user(user_id, item_id, user_sim_matrix, user_item_matrix)
    predictions.append(pred_rating)

#convert list to numpy array
predictions_array = np.array(predictions).reshape(-1, 1)
#create results array in correct submission format
results = np.hstack((test_data[:,:2], predictions_array, test_data[:,2:]))


#save results to csv: Int, Int, Float, Int
np.savetxt("results1.csv", results, delimiter=',', fmt=['%d', '%d', '%.1f', '%d'])


