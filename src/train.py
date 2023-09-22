import heapq
import numpy as np
import pandas as pd
from .make_dataset import load_dataset
from .helper import  get_train_instances, evaluate, random_mini_batches, preprocess
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tqdm import tqdm
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model:
    def __init__(self):
        self.num_neg = 4
        self.epochs = 20
        self.batch_size = 256
        self.learning_rate = 0.001
        self.latent_features = 8


    def predict(self, idx, num_of_products):
        map_product_score = {}
        idx=1
        K=10
        test_u = self.df_test['user_id'].values.tolist()
        test_i = self.df_test['product_id'].values.tolist()

        test_ratings = list(zip(test_u, test_i))

        self.df_neg =self. df_neg.drop(self.df_neg.columns[0], axis=1)
        test_negatives = self.df_neg.values.tolist()
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
        predictions = self.session.run([self.output_layer], feed_dict={self.user: predict_user, self.product: np_products})

        # Get the predicted scores as a list
        predictions = predictions[0].flatten().tolist()



        for i in range(len(products)):
            current_product = products[i]
     
            map_product_score[current_product] = predictions[i]

        # Get the K highest ranked products as a list
        k_ranked = heapq.nlargest(K, map_product_score, key=map_product_score.get)

        # recommendations = []
        # for i in k_ranked:
        #     recommendations.append((i, map_product_score[i]))
        return k_ranked


    def load_model(self, path, train=False):
        self.df = load_dataset()
        if os.path.exists(path) and not train:
            session = tf.Session()
            new_saver = tf.train.import_meta_graph(path + "/my-model.meta")
            new_saver.restore(session, tf.train.latest_checkpoint(path))
            all_vars = tf.get_collection('vars')
            for v in all_vars:
                v_ = session.run(v)
                print(v_)
        else:
            session = self.train_model()

        return session


    def train_model(self):
        # Load and prepare our data.

        self.all_user_ids, self.all_product_ids, self.df_train, self.df_test, self.df_neg, self.users, self.products, product_lookup = preprocess(self.df)
        
        self.MLP_model()
        self.GMP_model()
        session = self.NeuMF_model()
        return session


    def MLP_model(self):
        #-------------------------
        # TENSORFLOW GRAPH
        #-------------------------

        # Set up our Tensorflow graph
        graph = tf.Graph()

        with graph.as_default():

            # Define input placeholders for user, product and label.
            user = tf.placeholder(tf.int32, shape=(None, 1))
            product = tf.placeholder(tf.int32, shape=(None, 1))
            label = tf.placeholder(tf.int32, shape=(None, 1))

            # User feature embedding
            u_var = tf.Variable(tf.random_normal([len(self.users), 32], stddev=0.05), name='user_embedding')
            user_embedding = tf.nn.embedding_lookup(u_var, user)

            # product feature embedding
            i_var = tf.Variable(tf.random_normal([len(self.products), 32], stddev=0.05), name='product_embedding')
            product_embedding = tf.nn.embedding_lookup(i_var, product)

            # Flatten our user and product embeddings.
            user_embedding = tf.keras.layers.Flatten()(user_embedding)
            product_embedding = tf.keras.layers.Flatten()(product_embedding)

            # Concatenate our two embedding vectors together
            concatenated = tf.keras.layers.concatenate([user_embedding, product_embedding])

            # Add a first dropout layer.
            dropout = tf.keras.layers.Dropout(0.2)(concatenated)

            # Below we add our four hidden layers along with batch
            # normalization and dropouts. We use relu as the activation function.
            layer_1 = tf.keras.layers.Dense(64, activation='relu', name='layer1')(dropout)
            batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')(layer_1)
            dropout1 = tf.keras.layers.Dropout(0.2, name='dropout1')(batch_norm1)

            layer_2 = tf.keras.layers.Dense(32, activation='relu', name='layer2')(layer_1)
            batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm1')(layer_2)
            dropout2 = tf.keras.layers.Dropout(0.2, name='dropout1')(batch_norm2)

            layer_3 = tf.keras.layers.Dense(16, activation='relu', name='layer3')(layer_2)
            layer_4 = tf.keras.layers.Dense(8, activation='relu', name='layer4')(layer_3)

            # Our final single neuron output layer.
            self.output_layer = tf.keras.layers.Dense(1,
                    kernel_initializer="lecun_uniform",
                    name='output_layer')(layer_4)

            # Define our loss function as binary cross entropy.
            labels = tf.cast(label, tf.float32)
            logits = self.output_layer
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels,
                        logits=logits))

            # Train using the Adam optimizer to minimize our loss.
            opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            step = opt.minimize(loss)

            # Initialize all tensorflow variables.
            init = tf.global_variables_initializer()

        self.session = tf.Session(config=None, graph=graph)
        self.session.run(init)

        for epoch in range(self.epochs):

            # Get our training input.
            user_input, product_input, labels = get_train_instances(self.all_user_ids, self.all_product_ids, self.num_neg, self.products)

            # Generate a list of minibatches.
            minibatches = random_mini_batches(user_input, product_input, labels)

            # This has noting to do with tensorflow but gives
            # us a nice progress bar for the training
            progress = tqdm(total=len(minibatches))

            # Loop over each batch and feed our users, products and labels
            # into our graph. 
            for minibatch in minibatches:
                feed_dict = {user: np.array(minibatch[0]).reshape(-1,1),
                            product: np.array(minibatch[1]).reshape(-1,1),
                            label: np.array(minibatch[2]).reshape(-1,1)}
        
                # Execute the graph.
                _, l = self.session.run([step, loss], feed_dict)

                # Update the progress
                progress.update(1)
                progress.set_description('Epoch: %d - Loss: %.3f' % (epoch+1, l))

            progress.close()


        # Calculate top@K    
        hits = evaluate(self.df_neg, self.df_test, self.session, self.output_layer, user, product)
        print(np.array(hits).mean())


    def GMP_model(self):
        #-------------------------
        # TENSORFLOW GRAPH
        #-------------------------

        graph = tf.Graph()

        with graph.as_default():

            # Define input placeholders for user, product and label.
            user = tf.placeholder(tf.int32, shape=(None, 1))
            product = tf.placeholder(tf.int32, shape=(None, 1))
            label = tf.placeholder(tf.int32, shape=(None, 1))

            # User feature embedding
            u_var = tf.Variable(tf.random_normal([len(self.users), self.latent_features],
                                                stddev=0.05), name='user_embedding')
            user_embedding = tf.nn.embedding_lookup(u_var, user)

            # product feature embedding
            i_var = tf.Variable(tf.random_normal([len(self.products), self.latent_features],
                                                stddev=0.05), name='product_embedding')
            product_embedding = tf.nn.embedding_lookup(i_var, product)
            
            # Flatten our user and product embeddings.
            user_embedding = tf.keras.layers.Flatten()(user_embedding)
            product_embedding = tf.keras.layers.Flatten()(product_embedding)

            # Multiplying our user and product latent space vectors together 
            prediction_matrix = tf.multiply(user_embedding, product_embedding)

            # Our single neuron output layer
            self.output_layer = tf.keras.layers.Dense(1, 
                    kernel_initializer="lecun_uniform",
                    name='output_layer')(prediction_matrix)

            # Our loss function as a binary cross entropy. 
            loss = tf.losses.sigmoid_cross_entropy(label, self.output_layer)
            
            # Train using the Adam optimizer to minimize our loss.
            opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            step = opt.minimize(loss)

            # Initialize all tensorflow variables.
            init = tf.global_variables_initializer()

        self.session = tf.Session(config=None, graph=graph)
        self.session.run(init)


    def NeuMF_model(self):

        #-------------------------
        # TENSORFLOW GRAPH
        #-------------------------

        graph = tf.Graph()

        with graph.as_default():

            # Define input placeholders for user, product and label.
            self.user = tf.placeholder(tf.int32, shape=(None, 1))
            self.product = tf.placeholder(tf.int32, shape=(None, 1))
            label = tf.placeholder(tf.int32, shape=(None, 1))

            # User embedding for MLP
            mlp_u_var = tf.Variable(tf.random_normal([len(self.users), 32], stddev=0.05),
                    name='mlp_user_embedding')
            mlp_user_embedding = tf.nn.embedding_lookup(mlp_u_var, self.user)

            # product embedding for MLP
            mlp_i_var = tf.Variable(tf.random_normal([len(self.products), 32], stddev=0.05),
                    name='mlp_product_embedding')
            mlp_product_embedding = tf.nn.embedding_lookup(mlp_i_var, self.product)

            # User embedding for GMF
            gmf_u_var = tf.Variable(tf.random_normal([len(self.users), self.latent_features],
                stddev=0.05), name='gmf_user_embedding')
            gmf_user_embedding = tf.nn.embedding_lookup(gmf_u_var, self.user)

            # product embedding for GMF
            gmf_i_var = tf.Variable(tf.random_normal([len(self.products), self.latent_features],
                stddev=0.05), name='gmf_product_embedding')
            gmf_product_embedding = tf.nn.embedding_lookup(gmf_i_var, self.product)

            # Our GMF layers
            gmf_user_embed = tf.keras.layers.Flatten()(gmf_user_embedding)
            gmf_product_embed = tf.keras.layers.Flatten()(gmf_product_embedding)
            gmf_matrix = tf.multiply(gmf_user_embed, gmf_product_embed)

            # Our MLP layers
            mlp_user_embed = tf.keras.layers.Flatten()(mlp_user_embedding)
            mlp_product_embed = tf.keras.layers.Flatten()(mlp_product_embedding)
            mlp_concat = tf.keras.layers.concatenate([mlp_user_embed, mlp_product_embed])

            mlp_dropout = tf.keras.layers.Dropout(0.2)(mlp_concat)

            mlp_layer_1 = tf.keras.layers.Dense(64, activation='relu', name='layer1')(mlp_dropout)
            mlp_batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mlp_layer_1)
            mlp_dropout1 = tf.keras.layers.Dropout(0.2, name='dropout1')(mlp_batch_norm1)

            mlp_layer_2 = tf.keras.layers.Dense(32, activation='relu', name='layer2')(mlp_dropout1)
            mlp_batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mlp_layer_2)
            mlp_dropout2 = tf.keras.layers.Dropout(0.2, name='dropout1')(mlp_batch_norm2)

            mlp_layer_3 = tf.keras.layers.Dense(16, activation='relu', name='layer3')(mlp_dropout2)
            mlp_layer_4 = tf.keras.layers.Dense(8, activation='relu', name='layer4')(mlp_layer_3)

            # We merge the two networks together
            merged_vector = tf.keras.layers.concatenate([gmf_matrix, mlp_layer_4])

            # Our final single neuron output layer. 
            self.output_layer = tf.keras.layers.Dense(1,
                    kernel_initializer="lecun_uniform",
                    name='output_layer')(merged_vector)

            # Our loss function as a binary cross entropy. 
            loss = tf.losses.sigmoid_cross_entropy(label, self.output_layer)

            # Train using the Adam optimizer to minimize our loss.
            opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            step = opt.minimize(loss)

            # Initialize all tensorflow variables.
            init = tf.global_variables_initializer()


        self.session = tf.Session(config=None, graph=graph)
        self.session.run(init)

if __name__ == "__main__":
    a = Model()
    a.load_model("./models/model-1", train=True)
    print(a.predict(1, 20))