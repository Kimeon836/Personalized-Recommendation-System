"""
Some helper functions to train model

"""
import heapq
import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def get_negatives(all_user_ids, all_product_ids, products, df_test):
    """Returns a pandas dataframe of 100 negative interactions
    based for each user in df_test.

    Params:
        all_user_ids (np.array): Numpy array of all user ids.
        all_product_ids (np.array): Numpy array of all product ids.
        productsList: List of all unique products.
        df_test (dataframe): Our test set.
        
    Returns:
        df_neg (dataframe): dataframe with 100 negative products 
            for each (u, i) pair in df_test.
    """

    negativeList = []
    test_u = df_test['user_id'].values.tolist()
    test_i = df_test['product_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))
    zipped = set(zip(all_user_ids, all_product_ids))

    for (u, i) in test_ratings:
        negatives = []
        negatives.append((u, i))
        for t in range(100):
            j = np.random.randint(len(products)) # Get random product id.
            while (u, j) in zipped: # Check if there is an interaction
                j = np.random.randint(len(products)) # If yes, generate a new product id
            negatives.append(j) # Once a negative interaction is found we add it.
        negativeList.append(negatives)

    df_neg = pd.DataFrame(negativeList)

    return df_neg


def mask_first(x):
    """
    Return a list of 0 for the first product and 1 for all others
    """
    result = np.ones_like(x)
    result[0] = 0
    
    return result
   

def train_test_split(df):
    """
    Splits our original data into one test and one
    training set. 
    The test set is made up of one product for each user. This is
    our holdout product used to compute Top@K later.
    The training set is the same as our original data but
    without any of the holdout products.
    Params:
        df (dataframe): Our original data
    Returns:
        df_train (dataframe): All of our data except holdout products
        df_test (dataframe): Only our holdout products.
    """

    # Create two copies of our dataframe that we can modify
    df_test = df.copy(deep=True)
    df_train = df.copy(deep=True)

    # Group by user_id and select only the first product for
    # each user (our holdout).
    df_test = df_test.groupby(['user_id']).first()
    df_test['user_id'] = df_test.index
    df_test = df_test[['user_id', 'product_id', 'rating']]
    # del df_test.index.name

    # Remove the same products as we for our test set in our training set.
    mask = df.groupby(['user_id'])['user_id'].transform(mask_first).astype(bool)
    df_train = df.loc[mask]

    return df_train, df_test


def preprocess(df):
    df = df[["user_id", "product_id", "rating"]]

    df.columns = ['user', 'product', 'rating']
    df = df.dropna()
    df = df.loc[df.rating != 0]

    # Remove any users with fewer than 1 interaction. 
    # df_count = df.groupby(['user']).count()
    df['count'] = df.groupby('user')['user'].transform('count')
    df = df[df['count'] > 1]

    # Convert artists names into numerical IDs
    df['user_id'] = df['user'].astype("category").cat.codes
    df['product_id'] = df['product'].astype("category").cat.codes

    # Create a lookup frame so we can get the artist
    # names back in readable form later.
    product_lookup = df[['product_id', 'product']].drop_duplicates()
    product_lookup['product_id'] = product_lookup.product_id.astype(str)

    # Grab the columns we need in the order we need them.
    df = df[['user_id', 'product_id', 'rating']]

    # Create training and test sets.
    df_train, df_test = train_test_split(df)

    # Create lists of all unique users and artists
    users = list(np.sort(df.user_id.unique()))
    products = list(np.sort(df.product_id.unique()))

    # Get the rows, columns and values for our matrix.
    rows = df_train.user_id.astype(int)
    cols = df_train.product_id.astype(int)

    values = list(df_train.rating)

    # Get all unique user ids and product ids.
    all_user_ids = np.array(rows.tolist())
    all_product_ids = np.array(cols.tolist())

    # Sample 100 negative interactions for each user in our test data
    df_neg = get_negatives(all_user_ids, all_product_ids, products, df_test)

    return all_user_ids, all_product_ids, df_train, df_test, df_neg, users, products, product_lookup


def get_train_instances(all_user_ids, all_product_ids, num_neg, products):
     """Samples a number of negative user-product interactions for each
     user-product pair in our testing data.
     Returns:
         user_input -> List: A list of all users for each product
         product_input -> List: A list of all products for every user,
             both positive and negative interactions.
         labels -> List: A list of all labels. 0 or 1.
     """

     user_input, product_input, labels = [],[],[]
     zipped = set(zip(all_user_ids, all_product_ids))

     for (u, i) in zip(all_user_ids,all_product_ids):
         # Add our positive interaction
         user_input.append(u)
         product_input.append(i)
         labels.append(1)

         # Sample a number of random negative interactions
         for t in range(num_neg):
             j = np.random.randint(len(products))
             while (u, j) in zipped:
                 j = np.random.randint(len(products))
             user_input.append(u)
             product_input.append(j)
             labels.append(0)

     return user_input, product_input, labels


def random_mini_batches(U, I, L, mini_batch_size=256):
    """Returns a list of shuffeled mini batched of a given size.
    Params:
        U -> List: All users for every interaction 
        I -> List: All products for every interaction
        L -> List: All labels for every interaction.
    
    Returns:
        mini_batches -> List: A list of minibatches containing sets
            of batch users, batch products and batch labels 
            [(u, i, l), (u, i, l) ...]
    """

    mini_batches = []

    shuffled_U, shuffled_I, shuffled_L = shuffle(U, I, L)

    num_complete_batches = int(math.floor(len(U)/mini_batch_size))
    for k in range(0, num_complete_batches):
        mini_batch_U = shuffled_U[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_I = shuffled_I[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        mini_batch = (mini_batch_U, mini_batch_I, mini_batch_L)
        mini_batches.append(mini_batch)

    if len(U) % mini_batch_size != 0:
        mini_batch_U = shuffled_U[num_complete_batches * mini_batch_size: len(U)]
        mini_batch_I = shuffled_I[num_complete_batches * mini_batch_size: len(U)]
        mini_batch_L = shuffled_L[num_complete_batches * mini_batch_size: len(U)]

        mini_batch = (mini_batch_U, mini_batch_I, mini_batch_L)
        mini_batches.append(mini_batch)

    return mini_batches


def get_hits(k_ranked, holdout):
    """Return 1 if an product exists in a given list and 0 if not. """

    for product in k_ranked:
        if product == holdout:
            return 1
    return 0


def eval_rating(idx, test_ratings, test_negatives, K, session, output_layer, user, product):
    """Generate ratings for the users in our test set and
    check if our holdout product is among the top K highest scores.
    Params:
        idx (int): Current index
        test_ratings -> List: Our test set user-product pairs
        test_negatives -> List: 100 negative products for each
            user in our test set.
        K (int): number of top recommendations
    Returns:
        hr -> List: A list of 1 if the holdout appeared in our
            top K predicted products. 0 if not.
    """

    map_product_score = {}

    # Get the negative interactions our user.
    products = test_negatives[idx]

    # Get the user idx.
    user_idx = test_ratings[idx][0]

    # Get the product idx, i.e. our holdout product.
    holdout = test_ratings[idx][1]

    # Add the holdout to the end of the negative interactions list.
    products.append(holdout)

    # Prepare our user and product arrays for tensorflow.
    predict_user = np.full(len(products), user_idx, dtype='int32').reshape(-1,1)
    np_products = np.array(products).reshape(-1,1)

    # Feed user and products into the TF graph .
    predictions = session.run([output_layer], feed_dict={user: predict_user, product: np_products})

    # Get the predicted scores as a list
    predictions = predictions[0].flatten().tolist()

    # Map predicted score to product id.
    for i in range(len(products)):
        current_product = products[i]
        map_product_score[current_product] = predictions[i]

    # Get the K highest ranked products as a list
    k_ranked = heapq.nlargest(K, map_product_score, key=map_product_score.get)

    # Get a list of hit or no hit.   
    hits = get_hits(k_ranked, holdout)

    return hits


def evaluate(df_neg, df_test, session, output_layer, user, product, K=10):
    """Calculate the top@K hit ratio for our recommendations.
    Params:
        df_neg (dataframe): dataframe containing our holdout products
            and 100 randomly sampled negative interactions for each
            (user, product) holdout pair.
        K (int): The 'K' number of ranked predictions we want
            our holdout product to be present in. 
    Returns:
        hits (list): list of "hits". 1 if the holdout was present in 
            the K highest ranked predictions. 0 if not. 
    """

    hits = []

    test_u = df_test['user_id'].values.tolist()
    test_i = df_test['product_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))

    df_neg = df_neg.drop(df_neg.columns[0], axis=1)
    test_negatives = df_neg.values.tolist()

    for idx in range(len(test_ratings)):
        # For each idx, call eval_one_rating
        hitrate = eval_rating(idx, test_ratings, test_negatives, K, session, output_layer, user, product)
        hits.append(hitrate)

    return hits