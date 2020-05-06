"""
Submission Date: May 7, 2020
@author: Tomer Himi

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from time import time
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors


def plot_3d_scatter(data, labels):
    '''the function creates plot_3d_scatter
    parm: data: a given data frame
    type: data: data frame
    parm: labels: labels of data frame
    type: labels: list
    return: None'''
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.scatter(data[:,1], data[:,0], data[:,2], c = labels)
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.scatter(data[:,0], data[:,1], data[:,2], c = labels)
    plt.show()
    
def main():
    #Reads Data
    data = pd.read_csv('C:/Users/tomer/desktop/diabetic_data.csv')
    print("Initial data shape:", data.shape)  
    print(data.dtypes)
    print(data.head(10))

    #MDS
    dbscan_data_check = data.copy()
    dbscan_data = dbscan_data_check[['admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
                                     'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
                                     'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']]
    t0 = time()
    mds = manifold.MDS(2, max_iter = 30, n_init = 1)
    trans_data = mds.fit_transform(dbscan_data)
    t1 = time()
    print("MDS: %.2g sec" % (t1 - t0))
    print("trans_data:", trans_data)
    trans_data = pd.DataFrame(trans_data)
    plt.figure(1)
    sns.scatterplot(x = 0, y = 1, data = trans_data, palette = sns.color_palette("hls", 2), alpha = 0.5, legend = "full")
    
    
    #Data Cleaning
    none_kinds = ['?', 'NULL', 'None']
    data = data.replace(to_replace = none_kinds, value = np.nan) #replace Null strings to numpy NaN object

    #replace NULL ids according to mappings desciription 
    admission_type_id_null_values = [5, 6, 8]
    discharge_disposition_id_null_values = [18, 25, 26]
    admission_source_id_null_values = [9, 17, 20, 21]
    data['admission_type_id'] = data['admission_type_id'].replace(to_replace = admission_type_id_null_values, value = np.nan)
    data['discharge_disposition_id'] = data['discharge_disposition_id'].replace(to_replace = discharge_disposition_id_null_values, value = np.nan)
    data['admission_source_id'] = data['admission_source_id'].replace(to_replace = admission_source_id_null_values, value = np.nan)

    #drop every feature that as less than 80% data
    print(data.isnull().sum() / data.shape[0]) #check for data sparse precentage 
    features_to_remove = (data.isnull().sum() / data.shape[0]) > 0.2
    features_to_remove = features_to_remove[features_to_remove].index.values
    print("Removed features:", features_to_remove)
    data.drop(features_to_remove, axis = 1, inplace = True)

    #remove Nan
    data[['admission_type_id', 'discharge_disposition_id', 'admission_source_id']] = data[['admission_type_id', 'discharge_disposition_id', 'admission_source_id']].fillna(0)
    data['race'] = data['race'].fillna('Unknown')
    print("percentage of Nan's according the entire data:", data.isnull().values.ravel().sum() / data.shape[0]) 
    data = data.dropna()
    #remove Unknown/Invalid
    data = data[data['gender'] != 'Unknown/Invalid']
    print("Data shape after cleaning step:", data.shape)

    #handle data duplications of records
    print(len(data['patient_nbr'].unique()), len(data['encounter_id'].unique()))
    data = data.drop_duplicates(subset = 'patient_nbr', keep = 'first')

    
    #Statistics Of The Data Set
    print(data.groupby('gender').size())
    plt.figure(2)
    pd.crosstab(data['race'], data['gender']).plot(kind = 'bar', title = 'Race & Gender', color = ['blue','cyan'])
    plt.figure(4)
    ax = data.groupby('age').size().plot(kind = 'bar',title = 'Age')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
    data.groupby('age').size().plot.line(color = 'red')
    plt.show()
    plt.figure(5)
    pd.crosstab(data['age'], data['readmitted']).plot(kind = 'bar', title = 'Age & Readmitted', color = ['red', 'blue', 'green'])
    print(pd.crosstab(data['race'], data['readmitted']))
    plt.figure(7)
    ax = data.groupby('readmitted').size().plot(kind = 'bar', title = 'Readmitted')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x / data.shape[0]) for x in vals])
    print(ax)
    plt.figure(8)
    pd.crosstab(data['time_in_hospital'], data['readmitted']).plot(kind = 'bar', color = ['red', 'blue', 'green'])
    #plt.figure(10)
    #pd.crosstab(data['readmitted'], data['change']).plot(kind = 'bar', title = 'Change & Readmitted', color = ['green', 'red'])
    print(pd.crosstab(data['age'], data['diabetesMed']))
    print(data['time_in_hospital'].describe())
    plt.figure(10)
    data['num_medications'].plot.hist(bins = 50, histtype = 'stepfilled', color = 'pink', title = 'num of medications')
    plt.show()
    
    
    #Preprocessing The Data 
    categorical_data = data.select_dtypes('object')
    numeric_data = data[list(filter(lambda col: col not in categorical_data.columns, data.columns))]

    #handle categorical data
    drugs_data = categorical_data[['metformin', 'citoglipton', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']]
    for col in drugs_data.columns:
        drugs_data[col] = drugs_data[col].apply(lambda val: 0 if val == 'No' or val == 'Steady' else 1)
    categorical_data['number_drugs_changed'] = drugs_data.sum(axis = 1)
    categorical_data = categorical_data.drop(drugs_data.columns, axis = 1)
    
    categorical_data['gender'] = categorical_data['gender'].replace('Male', 1)
    categorical_data['gender'] = categorical_data['gender'].replace('Female', 0)
    categorical_data['change'] = categorical_data['change'].replace('Ch', 1)
    categorical_data['change'] = categorical_data['change'].replace('No', 0)
    categorical_data['diabetesMed'] = categorical_data['diabetesMed'].replace('Yes', 1)
    categorical_data['diabetesMed'] = categorical_data['diabetesMed'].replace('No', 0)
    categorical_data['readmitted'] = categorical_data['readmitted'].replace('NO', 0)
    categorical_data['readmitted'] = categorical_data['readmitted'].replace('>30', 0)
    categorical_data['readmitted'] = categorical_data['readmitted'].replace('<30', 1)
    categorical_data['age'] = categorical_data['age'].replace('[0-10)', 5)
    categorical_data['age'] = categorical_data['age'].replace('[10-20)', 15)
    categorical_data['age'] = categorical_data['age'].replace('[20-30)', 25)
    categorical_data['age'] = categorical_data['age'].replace('[30-40)', 35)
    categorical_data['age'] = categorical_data['age'].replace('[40-50)', 45)
    categorical_data['age'] = categorical_data['age'].replace('[50-60)', 55)
    categorical_data['age'] = categorical_data['age'].replace('[60-70)', 65)
    categorical_data['age'] = categorical_data['age'].replace('[70-80)', 75)
    categorical_data['age'] = categorical_data['age'].replace('[80-90)', 85)
    categorical_data['age'] = categorical_data['age'].replace('[90-100)', 95)
   
    race_dummies_data = pd.get_dummies(categorical_data['race'])
    categorical_data = categorical_data.drop('race', axis = 1)
    categorical_data = pd.concat([categorical_data, race_dummies_data], axis = 1)

    #handle numeric data
    #combining the 'admission_type_id' categories 2->1 and 7->1
    numeric_data['admission_type_id'] = numeric_data['admission_type_id'].replace(2, 1)
    numeric_data['admission_type_id'] = numeric_data['admission_type_id'].replace(7, 1)
    #reanme numbers to categories
    numeric_data['admission_type_id'] = numeric_data['admission_type_id'].replace(0, 'Not Available')
    numeric_data['admission_type_id'] = numeric_data['admission_type_id'].replace(1, 'Emergency')
    numeric_data['admission_type_id'] = numeric_data['admission_type_id'].replace(3, 'Elective')
    numeric_data['admission_type_id'] = numeric_data['admission_type_id'].replace(4, 'Newborn')
    numeric_data = pd.get_dummies(numeric_data, 'admission_type_id')
    numeric_data = numeric_data.drop(['discharge_disposition_id', 'admission_source_id'], axis = 1)
    combined_data = pd.concat([numeric_data, categorical_data], axis = 1)
    combined_data = combined_data.drop(['diag_1', 'diag_2', 'diag_3'], axis = 1)
    combined_data = combined_data.set_index(['encounter_id', 'patient_nbr'])

    #using standard scaler
    numeric_not_binary_columns = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'number_drugs_changed']
    scaler = StandardScaler()
    scaler.fit(combined_data[numeric_not_binary_columns])
    combined_data[numeric_not_binary_columns] = scaler.transform(combined_data[numeric_not_binary_columns])
    
    f, ax = plt.subplots(figsize = (10, 8))
    corr = combined_data.corr()
    sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), cmap = sns.diverging_palette(10, 145, as_cmap = True), square = True, ax = ax)
    #combined_data.hist()  #histograms of every colomn after pre-processing
    
    #Clustering Part
    #K-Means Clustering
    silhouette_scores = {}
    elbow_scores = {}
    for k in range(2,9):
        kmeans_model = KMeans(n_clusters = k, n_jobs = -1)
        kmeans_model.fit(combined_data.values)
        elbow_scores.update({k:kmeans_model.inertia_})
        silhouette_score_val = silhouette_score(combined_data, kmeans_model.labels_)
        silhouette_scores.update({k:silhouette_score_val})
    #Sןilouhette score
    plt.figure(figsize=(7,4))
    plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
    plt.scatter(x=list(silhouette_scores.keys()),y=list(silhouette_scores.values()),s=150,edgecolor='k')
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("Silhouette score",fontsize=15)
    plt.xticks([i for i in range(2,12)],fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.title('The Elbow Method using Inertia') 
    plt.plot(list(elbow_scores.keys()),list(elbow_scores.values()), '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')

    #run K-Means algorithm
    kmeans_model = KMeans(n_clusters = 6, n_jobs = -1)
    kmeans_model.fit(combined_data.values)
    #perform PCA for plotting
    kmeans_pca = PCA(n_components = 3)
    kmeans_pca_data = kmeans_pca.fit_transform(combined_data)
    plot_3d_scatter(kmeans_pca_data, kmeans_model.labels_)
    analysis_data = combined_data.copy()
    analysis_data['race'] = analysis_data[['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other', 'Unknown']].idxmax(axis = 1)
    analysis_data['admission_type_id'] = analysis_data[['admission_type_id_Elective', 'admission_type_id_Emergency', 'admission_type_id_Newborn', 'admission_type_id_Not Available']].idxmax(axis = 1)
    analysis_data[numeric_not_binary_columns] = scaler.inverse_transform(analysis_data[numeric_not_binary_columns])
    analysis_data['labels'] = kmeans_model.labels_
    #show means for numeric features 
    print(analysis_data[numeric_not_binary_columns + ['labels']].groupby('labels').mean())
    print(pd.crosstab(analysis_data['labels'], analysis_data['race']))
    
    #Agglomerative Clustering
    silhouette_scores = {}
    for k in range(2,9):
        knn = kneighbors_graph(combined_data, 10, n_jobs = -1)
        Agglomerative_model = AgglomerativeClustering(n_clusters = k, connectivity = knn)
        Agglomerative_model.fit(combined_data)
        silhouette_score_val = silhouette_score(combined_data, Agglomerative_model.labels_)
        silhouette_scores.update({k:silhouette_score_val})
    plt.figure(11)
    plt.bar(list(silhouette_scores.keys()), list(silhouette_scores.values()), align = 'center', alpha = 0.5)
    plt.xlabel("Number of clusters")
    plt.ylabel("silhouette_scores")
    plt.title('silhouette_scores of Agglomerative Clustering')
    plt.plot()
    plt.show()

    #run Agglomerative algorithm
    knn = kneighbors_graph(combined_data, 10, n_jobs = -1)
    Agglomerative_model = AgglomerativeClustering(n_clusters = 4, connectivity = knn)
    Agglomerative_model.fit(combined_data)
    #perform PCA for plotting
    agglomerative_pca = PCA(n_components = 3)
    agglomerative_pca_data = agglomerative_pca.fit_transform(combined_data)
    plot_3d_scatter(agglomerative_pca_data, Agglomerative_model.labels_)
    analysis_data['labels'] = Agglomerative_model.labels_
    #show means for numeric features 
    print(analysis_data[numeric_not_binary_columns + ['labels']].groupby('labels').mean())
    plt.figure(12)
    pd.crosstab(analysis_data['labels'],analysis_data['race']).plot(kind='bar')
    print(pd.crosstab(analysis_data['labels'],analysis_data['change']))
    print(pd.crosstab(analysis_data['labels'],analysis_data['diabetesMed']))

    #Gaussian Mixture Clustering
    gm_bic= []
    for i in range(2,9):
        gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(combined_data)
        gaussian_mixture_model_labels = gm.fit_predict(combined_data)
        gm_bic.append(-gm.bic(combined_data))
    plt.figure(figsize=(6,3))
    plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
    plt.scatter(x=[i for i in range(2,9)],y=np.log(gm_bic),s=150,edgecolor='k')
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
    plt.xticks([i for i in range(2,9)],fontsize=14)
    plt.yticks(fontsize=12)
    plt.show()
    #v_measure_scores = {}
    for i in range(2,9): 
        gm = GaussianMixture(n_components = i, n_init = 10, tol = 1e-3, max_iter = 1000).fit(combined_data)
        gaussian_mixture_model_labels = gm.fit_predict(combined_data)
        #v_measure_val = v_measure_score(combined_data['readmitted'], gaussian_mixture_model_labels)
        #v_measure_scores.update({i:v_measure_val})
    
    #Gaussian Mixture
    silhouette_scores = {}
    for k in range(2,9):
        gaussian_mixture_model = GaussianMixture(n_components=k)
        gaussian_mixture_model.fit(combined_data.values)
        gaussian_mixture_model_labels = gaussian_mixture_model.fit_predict(combined_data.values)
        silhouette_score_val = silhouette_score(combined_data,gaussian_mixture_model_labels)
        silhouette_scores.update({k:silhouette_score_val})  
    plt.figure(13)
    plt.bar(list(silhouette_scores.keys()), list(silhouette_scores.values()), align = 'center', alpha = 0.5)
    plt.xlabel("Number of clusters")
    plt.ylabel("silhouette_scores")
    plt.title('silhouette_scores of GMM clustering')
    plt.plot()
    plt.show()

    #run Gaussian Mixture algorithm
    gaussian_mixture_model = GaussianMixture(n_components = 5).fit(combined_data)
    gaussian_mixture_model_labels = gaussian_mixture_model.fit_predict(combined_data) 
    #perform PCA for plotting
    gaussian_mixture_pca = PCA(n_components = 3)
    gaussian_mixture_pca_data = gaussian_mixture_pca.fit_transform(combined_data)
    plot_3d_scatter(gaussian_mixture_pca_data, gaussian_mixture_model_labels)
    analysis_data = combined_data.copy()
    analysis_data['labels'] = gaussian_mixture_model_labels
    #show means for numeric features 
    print(analysis_data[numeric_not_binary_columns + ['labels']].groupby('labels').mean())
    plt.figure(15)
    print(pd.crosstab(analysis_data['labels'],analysis_data['race']).plot(kind='bar'))
    
    #DBSCAN Clustering
    #KNN algorithm
    neigh = NearestNeighbors(n_neighbors = 2)
    nbrs = neigh.fit(dbscan_data)
    distances, indices = nbrs.kneighbors(dbscan_data)
    distances = np.sort(distances, axis = 0)
    distances = distances[:,1]
    plt.figure(figsize=(8,4))
    plt.plot(distances)
    print('average:', np.mean(distances, axis = None, dtype = None, out = None))
    minsample = np.log(len(dbscan_data))
    print('minsample:', minsample)     
    plt.figure(19)
    plt.hist(distances)
    plt.show()
    
    #run DBSCAN algorithm
    db = DBSCAN(eps = 6.2 ,min_samples = 7.8)
    db_label = db.fit_predict(dbscan_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    #Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    centers = db.core_sample_indices_
    X = trans_data.loc[:, [0,1,'pred_labels']]
    X.plot.scatter(x = 0, y = 1, c = db_label, s = 50, cmap = 'viridis')
    plt.scatter(centers[0], centers[1], c = 'black', s = 200, alpha = 0.3)

if __name__ == "__main__":
    main()