
import yaml
import json
import time
import logging
import joblib
import argparse
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from data_engineering import DataAnalyzer, FeatureExtractor
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from problem_config import ProblemConfig, ProblemConst, get_prob_config
from raw_data_processor import *
from scipy.spatial import distance


class ClusteringEvaluator:
    def __init__(self, X = None, y = None, model = None, n_cluster = None):
        self.X = X
        self.y = y
        self.model = model
        self.n_cluster = n_cluster

    def evaluate_clustering(self):
        # Calculate Silhouette Coefficient
        logging.info("Calculate silhouette_score...")
        start_time = time.time()
        silhouette = metrics.silhouette_score(self.X, self.model.labels_)
        end_time = time.time()
        silhouette_time = end_time - start_time

        #Predict label again
        new_labels = []
        kmeans_clusters = self.model.predict(self.X)
        for cluster in range(self.n_cluster):
            mask = self.model.labels_ == cluster
            if len(self.y[mask]) == 0:
                # If no data points in the cluster, assign a default label (e.g., 0)
                new_labels.append(1)
                continue
            most_common_label = np.bincount(self.y[mask]).argmax()
            new_labels.append(most_common_label)
        approx_label = [new_labels[c] for c in kmeans_clusters]
        
        
        print('Truth labels : ', np.unique(self.y, return_counts= True))
        print('Predicted labels: ', np.unique(approx_label, return_counts=True))


        # Calculate Rand Index
        logging.info("Calculate rand_index...")
        start_time = time.time()
        rand_index = metrics.adjusted_rand_score(self.y, approx_label)
        end_time = time.time()
        rand_index_time = end_time - start_time

        # Calculate Sum of Squared Distance (SSD)
        logging.info('Calculate Sum of Squared Distance (SSD)')
        start_time = time.time()
        ssd = self.model.inertia_
        end_time = time.time()
        ssd_time = end_time - start_time

        # compute contingency matrix (also called confusion matrix)
        logging.info('Calculate purity score')
        start_time = time.time()
        contingency_matrix = metrics.cluster.contingency_matrix(self.y, approx_label)
        # calculate purity for each cluster
        purity = np.amax(contingency_matrix, axis=0) / np.sum(contingency_matrix, axis=0)
        end_time = time.time()
        purity_time = end_time - start_time

        logging.info(f'Silhouette Coefficient: {silhouette:.2f} (Elapsed time: {silhouette_time:.2f} seconds)')
        logging.info(f'Rand Index: {rand_index:.2f} (Elapsed time: {rand_index_time:.2f} seconds)')
        logging.info(f'Sum of Squared Distance (SSD): {ssd:.2f} (Elapsed time: {ssd_time:.2f} seconds)')
        logging.info(f'Avg Purity of clustering: {np.mean(purity):.2f} (Elapsed time: {purity_time:.2f} seconds)')

    # def DBSCAN_evaluate(self):
        
    #     # Calculate Silhouette Coefficient
    #     logging.info("Calculate silhouette_score...")
    #     start_time = time.time()
    #     silhouette = metrics.silhouette_score(self.X, self.model.labels_)
    #     end_time = time.time()
    #     silhouette_time = end_time - start_time

    #     #Predict label again

    #     approx_label = dbscan_predict(self.model, self.X)
    #     print('Truth labels : ', np.unique(self.y, return_counts= True))
    #     print('Predicted labels: ', np.unique(approx_label, return_counts=True))

    #     # Calculate Rand Index
    #     logging.info("Calculate rand_index...")
    #     start_time = time.time()
    #     rand_index = metrics.adjusted_rand_score(self.y, approx_label)
    #     end_time = time.time()
    #     rand_index_time = end_time - start_time

    #     # compute contingency matrix (also called confusion matrix)
    #     logging.info('Calculate purity score')
    #     start_time = time.time()
    #     contingency_matrix = metrics.cluster.contingency_matrix(self.y, approx_label)
    #     # calculate purity for each cluster
    #     purity = np.amax(contingency_matrix, axis=0) / np.sum(contingency_matrix, axis=0)
    #     end_time = time.time()
    #     purity_time = end_time - start_time

    #     logging.info(f'Silhouette Coefficient: {silhouette:.2f} (Elapsed time: {silhouette_time:.2f} seconds)')
    #     logging.info(f'Rand Index: {rand_index:.2f} (Elapsed time: {rand_index_time:.2f} seconds)')
    #     logging.info(f'Avg Purity of clustering: {np.mean(purity):.2f} (Elapsed time: {purity_time:.2f} seconds)')



# Define a function to assign new data points to clusters
# def dbscan_predict(dbscan_model, X_new, metric=distance.euclidean):
#     # Result is noise by default
#     y_new = np.ones(shape=len(X_new), dtype=int) #*-1

#     # Iterate all input samples for a label
#     for j, x_new in enumerate(X_new):
#         # Find a core sample closer than EPS
#         for i, x_core in enumerate(dbscan_model.components_):
#             if metric(x_new, x_core) < dbscan_model.eps:
#                 # Assign label of x_core to x_new
#                 y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
#                 break

#     return y_new

# def prob2_propagate_labels(labeled_data, labeled_labels, unlabeled_data):
#     #load config file
#     config_path = './src/model_config/'+ prob_config.phase_id + '/' + prob_config.prob_id +'/cluster.json'
#     with open(config_path, "r") as f:
#         model_params = json.load(f)

#     #define model
#     algorithm = model_params["algorithm"]["name"] # 'k-means' or 'DBSCAN' or 'MiniBatchKMeans'
#     logging.info(f"Use {algorithm} algorithm to labeling captured data")
#     logging.info(f"Parameters: {model_params['dbscan']}")
#     clusterer = DBSCAN(**model_params["dbscan"])

#     #train model
#     #Load if existed or train new model
#     saved_model = f'./src/model_config/{prob_config.phase_id }/{prob_config.prob_id }/{algorithm}_{model_params["dbscan"]["eps"]}_{model_params["dbscan"]["min_samples"]}_model.pkl'
#     if os.path.isfile(saved_model):
#         logging.info("Oh existed trained model, nice!")
#         clusterer = joblib.load(saved_model)
#     else:
#         logging.info("Ops, waiting for happiness :))")
#         start_time = time.time()
#         clusterer.fit(labeled_data)
#         end_time = time.time()
#         joblib.dump(clusterer, saved_model)
#         logging.info(f'Elapsed time: {(end_time - start_time):.2f}')
    
#     print(np.unique(clusterer.labels_, return_counts=True))
#     return

#     #evaluate
#     logging.info('Evaluate cluster model... ')
#     evaluator = ClusteringEvaluator(X=labeled_data, y=labeled_labels, model=clusterer)
#     evaluator.DBSCAN_evaluate()

#     # Step 3: Propagate labels to the rest of the data

#     logging.info("Labeling new data...")
#     propagated_labels = dbscan_predict(clusterer, unlabeled_data)

#     all_labels = np.concatenate((labeled_labels, propagated_labels), axis=0)
#     # Merge the labeled and unlabeled data
#     data = np.concatenate((labeled_data, unlabeled_data), axis=0)

#     return data, all_labels, unlabeled_data, propagated_labels

def asign_label(clusterer, n_cluster, labeled_labels):
    new_labels = []
    
    for cluster in range(n_cluster):
        mask = clusterer.labels_ == cluster
        if len(labeled_labels[mask]) == 0:
            new_labels.append(-1)
            continue
        most_common_label = np.bincount(labeled_labels[mask]).argmax()
        new_labels.append(most_common_label)
    
    # Get the centroids of each cluster
    centroids = clusterer.cluster_centers_
    # Set the batch size
    batch_size = 1000
    # Initialize an array to store the nearest cluster for each cluster
    nearest_clusters = np.empty(centroids.shape[0], dtype=int)
    # Calculate the distances between the centroids in batches
    for i in range(0, centroids.shape[0], batch_size):
        # Get the current batch of centroids
        batch = centroids[i:i+batch_size]
        
        # Calculate the distances between the current batch of centroids and all the other centroids
        distances = cdist(batch, centroids)
        
        # Set the diagonal to infinity to exclude the distance between a centroid and itself
        np.fill_diagonal(distances, np.inf)
        
        # Find the nearest cluster for each centroid in the current batch
        nearest_clusters[i:i+batch_size] = np.argmin(distances, axis=1)

    # print(f"The nearest clusters are: {nearest_clusters}")

    for cluster in tqdm(range(n_cluster), ncols=100, desc="label_: "):
        if new_labels[cluster] == -1:
            # Find the nearest cluster to the first cluster (cluster 0)
            # Asign label by the label of nearest cluster
            if new_labels[nearest_clusters[cluster]] != -1:
                new_labels[cluster]  = new_labels[nearest_clusters[cluster]]
            else:
                new_labels[cluster]  = 1


    return new_labels

def prob1_propagate_labels(labeled_data, labeled_labels, unlabeled_data):

    config_path = './src/model_config/'+ prob_config.phase_id + '/' + prob_config.prob_id +'/cluster.json'
    with open(config_path, "r") as f:
        model_params = json.load(f)

    algorithm = model_params["algorithm"]["name"] # 'k-means' or 'MiniBatchKMeans'
    logging.info(f"Use {algorithm} algorithm to labeling captured data")


    if algorithm == 'k-means':
        # Step 2: Cluster the data using k-means
        logging.info(f"Parameters: {model_params['k_means']}")
        clusterer = KMeans(**model_params["k_means"])
        n_cluster = model_params["k_means"]["n_clusters"]

    elif algorithm == 'MiniBatchKMeans':
        # Step 2: Cluster the data using MiniBatchKMeans
        logging.info(f"Parameters: {model_params['mini']}")
        clusterer = MiniBatchKMeans(**model_params["mini"])
        n_cluster = model_params["mini"]["n_clusters"]

    logging.info("Fitting labeled data...")

    saved_model = f'./src/model_config/{prob_config.phase_id }/{prob_config.prob_id }/{n_cluster}_{algorithm}_model.pkl'
    if os.path.isfile(saved_model):
        logging.info("Oh existed trained model, nice!")
        clusterer = joblib.load(saved_model)
        n_features = clusterer.cluster_centers_.shape[1]
    else:
        logging.info("Ops, waiting for happiness :))")

        start_time = time.time()
        clusterer.fit(labeled_data)
        end_time = time.time()
        joblib.dump(clusterer, saved_model)
        logging.info(f'Elapsed time: {(end_time - start_time):.2f}')
        n_features = clusterer.cluster_centers_.shape[1]
    

    # logging.info('Evaluate cluster model... ')
    # evaluator = ClusteringEvaluator(X=labeled_data[:, :n_features], y=labeled_labels, model=clusterer[:, :n_features], n_cluster=n_cluster)
    # evaluator.evaluate_clustering()

    # Step 3: Propagate labels to the rest of the data
    logging.info("Labeling new data...")
    
    kmeans_clusters = clusterer.predict(unlabeled_data[:, :n_features])
    new_labels = asign_label(clusterer, n_cluster, labeled_labels)
    propagated_labels = [new_labels[c] for c in kmeans_clusters]


    # Merge the labeled and unlabeled data
    all_labels = np.concatenate((labeled_labels, propagated_labels), axis=0)
    data = np.concatenate((labeled_data, unlabeled_data), axis=0)

    return data, all_labels, unlabeled_data, propagated_labels

def label_captured_data(prob_config: ProblemConfig):

    eda = DataAnalyzer(prob_config)
    eda.load_data()
    eda.preprocess_data(target_col = eda.target_col)
    if eda.prob_config.prob_id == 'prob-1':
        eda.prob1_process()
        # self.feature_selection(dev = 1)
    else:
        eda.prob2_process()
        #  self.feature_selection(dev = 5)
    training_data = eda.data

    data = training_data.drop([prob_config.target_col], axis=1)
    data_dtype = data.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()
    columns = data.columns
    labeled_data = data.to_numpy()

    # new = pd.read_parquet(prob_config.captured_x_path)
    # new = new[columns]
    # new.to_parquet(prob_config.captured_x_path, index=False)
    # return

    # for c, (k, v) in zip(columns, data_dtype.items()):
    #     print(c==k, v)
    # return

    labels = pd.DataFrame(training_data[prob_config.target_col])
    labeled_labels = labels.to_numpy().squeeze()
    labels_dtype = labels.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()


    ml_type = prob_config.ml_type
    # print(labeled_labels.squeeze().shape)

    logging.info("Load captured data")

    captured_x = pd.DataFrame()
    for file_path in tqdm(prob_config.captured_data_dir.glob("*.parquet"), ncols=100, desc ="Loading...", unit ="file"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])
        os.remove(file_path)

    captured_x.to_parquet(prob_config.captured_data_dir / "total_data.parquet")
    captured_x = eda.preprocess_data(input_data=captured_x)    

    
    logging.info('Preprocessing captured data....')
    if prob_config.prob_id == 'prob-1':
        path_save = "./src/model_config/phase-1/prob-1/sub_values_captured.pkl"

        if os.path.isfile(path_save):
            logging.info("Remove old file")
            os.remove(path_save)

        extractor = FeatureExtractor(captured_x, path_save)
        captured_x = extractor.create_new_feature(captured_x)

        unlabeled_data = captured_x[captured_x['is_drift']==1] #just use drift
        
        unlabeled_data = unlabeled_data[columns].to_numpy()

    else: 
        
        unlabeled_data = captured_x[captured_x['is_drift']==1] #just use drift
        unlabeled_data = captured_x[columns].to_numpy()

    n_captured = len(unlabeled_data)
    n_samples = len(labeled_data) + n_captured

    logging.info(f"Loaded {n_captured} captured samples")

    print('unlabled: ', unlabeled_data.shape)
    print('labled: ', labeled_data.shape)

    logging.info("Initialize and fit the clustering model")

    # if prob_config.prob_id == 'prob1':
    #     total_data, total_label, captured_data, approx_label = prob1_propagate_labels(labeled_data, labeled_labels, unlabeled_data)
    #     # print(np.unique(approx_label))
    # else:
    #     total_data, total_label, captured_data, approx_label = prob2_propagate_labels(labeled_data, labeled_labels, unlabeled_data)
    
    _, _, _, approx_label = prob1_propagate_labels(labeled_data, labeled_labels, unlabeled_data)


    logging.info("Saving new data...")
    captured_x = pd.DataFrame(unlabeled_data, columns = columns)

    for column in captured_x.columns:
        captured_x[column] = captured_x[column].astype(data_dtype[column])

    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])
    for column in approx_label_df.columns:
        approx_label_df[column] = approx_label_df[column].astype(labels_dtype[column])
                              
    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)
    print(captured_x.info(), '\n', approx_label_df.info(), "\n", np.unique(approx_label_df, return_counts=True))
    logging.info(f"----- After process have {len(captured_x)} labeled captured data ------")
    logging.info('Done!')

    # print(len(np.unique(pd.concat([captured_x['feature4'], captured_x['feature7']]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)

    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    
    # label_captured_data(prob_config, model_params)
    label_captured_data(prob_config)

