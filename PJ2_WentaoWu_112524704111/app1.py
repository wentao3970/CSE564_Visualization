import json
from sklearn import metrics, decomposition, manifold
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist

from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
#First of all you have to import it from the flask module:
app = Flask(__name__)
#By default, a route only answers to GET requests. You can use the methods argument of the route() decorator to handle different HTTP methods.
@app.route("/", methods = ['POST', 'GET'])
def index():
    global df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df
    #The current request method is available by using the method attribute
    if request.method == 'POST':
        if request.form['data'] == 'ran_df':
            data = y_ran
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

        if request.form['data'] == 'raw_df':
            data = y_raw
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

        if request.form['data'] == 'y_str':
            data = y_str
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

    data = y_str
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html", data=data)


#Two PCA vectors ploting
@app.route("/2dplot", methods = ['POST', 'GET'])
def _2dplot():
    global df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_str)
    principal2_df = pd.DataFrame(data=principalComponents,columns=['PC1','PC2'])
    principal2_df['price'] = target

    data = principal2_df
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("2dplot.html", data=data)


#MDS 2D Plots using Euclidean and correlation distances
@app.route("/mds", methods = ['POST', 'GET'])
def mds():
    global df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df_e,mds_df_c
    if request.method == 'POST':
        if request.form['data'] == 'mds_df_c':
            data = mds_df_c
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)
        if request.form['data'] == 'mds_df_e':
            data = mds_df_e
            chart_data = data.to_dict(orient='records')
            chart_data = json.dumps(chart_data, indent=2)
            data = {'chart_data': chart_data}
            return jsonify(data)

    #(1)Euclidean distance
    #return a dataframe
    mds_df_e = pd.DataFrame()
    mds_e = MDS(n_components=2, dissimilarity='euclidean')
    mds_e = mds_e.fit_transform(x_str)
    mds_e = pd.DataFrame(mds_e)
    mds_df_e['x'] = mds_e[0]
    mds_df_e['y'] = mds_e[1]
    mds_df_e['price'] = target

    #(2)Correlation distance
    #return a dataframe
    x_str_df = pd.DataFrame(x_str)
    x_str_df = x_str_df.transpose()
    cor_matrix = x_str_df.corr()
    for col in cor_matrix.columns:
        cor_matrix[col].values[:] = 1 - cor_matrix[col].values[:]
    mds_c = MDS(n_components=2, dissimilarity='precomputed')
    mds_df_c = mds_c.fit_transform(cor_matrix)
    mds_df_c = pd.DataFrame(mds_df_c)
    mds_df_c['x'] = mds_df_c[0]
    mds_df_c['y'] = mds_df_c[1]
    mds_df_c['price'] = target

    data = mds_df_e
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("mds.html", data=data)


#Three highest loading attributes plot matrix
@app.route("/attri_matrix", methods = ['POST', 'GET'])
def attri_matrix():
    global df,dataClean,no_ch_df,strati_df,y_raw,y_str,y_ran,mds_df
    mainAttri_df = strati_df[['MedianListingPrice_SingleFamilyResidence','MedianListingPrice_2Bedroom','MedianListingPrice_3Bedroom','price']]

    data = mainAttri_df
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("attri_matrix.html", data=data)


if __name__ == "__main__":
    df = pd.read_csv('AmericanHousingPrice.csv')
    dataClean = df.fillna(df.mean())
    no_ch_df = dataClean.drop(['Date','RegionName'], axis=1)

    #Random sampling
    random_df = dataClean.sample(frac=0.5)

    #Kmeans clustering, Elbow has found the optimal k is 5
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(no_ch_df)
        kmeanModel.fit(no_ch_df)
        distortions.append(sum(np.min(cdist(no_ch_df,kmeanModel.cluster_centers_,
        'euclidean'),axis=1))/no_ch_df.shape[0])

    kmeans=KMeans(n_clusters=5, max_iter=500)
    kmeans.fit(no_ch_df)

    strati_df = pd.DataFrame()
    n_clusters = 5
    persnt = 0.5
    for i in range(n_clusters):
        Clstr_i = np.where(kmeans.labels_ == i)[0].tolist()
        num_i = len(Clstr_i)
        sample_i = np.random.choice(Clstr_i, int(persnt*num_i))
        i_cluster_df = no_ch_df.loc[sample_i]
        strati_df = pd.concat([strati_df,i_cluster_df],axis = 0)

    #PCA
    #devide the midean housing price into 5 groups according to the price level
    def function(a):
        if a < 180000: return 'very low-price housing'
        if a>=180000 and a< 250000: return 'low-price housing'
        if a>=250000 and a< 380000: return 'medium-price housing'
        if a>=380000 and a< 520000: return 'high-price housing'
        else: return 'very high-price housing'
    #add the target column to the stritified data frame
    strati_df['price'] = strati_df.apply(lambda x: function(x.MedianListingPrice_AllHomes), axis = 1)
    target = strati_df.loc[:,['price']].values

    rawdat_df2 = dataClean.drop(['Date','RegionName','MedianListingPrice_AllHomes'],axis=1)
    strati_df2 = strati_df.drop(['MedianListingPrice_AllHomes','price'],axis=1)
    random_df2 = random_df.drop(['Date','RegionName','MedianListingPrice_AllHomes'],axis=1)
    x_raw = StandardScaler().fit_transform(rawdat_df2)
    x_str = StandardScaler().fit_transform(strati_df2)
    x_ran = StandardScaler().fit_transform(random_df2)
    rawdat_pca = decomposition.PCA()
    strati_pca = decomposition.PCA()
    random_pca = decomposition.PCA()
    rawdat_pca.fit(x_raw)
    strati_pca.fit(x_str)
    random_pca.fit(x_ran)
    y_raw = pd.DataFrame()
    y_str = pd.DataFrame()
    y_ran = pd.DataFrame()
    y_raw['variance'] = rawdat_pca.explained_variance_
    y_str['variance'] = strati_pca.explained_variance_
    y_ran['variance'] = random_pca.explained_variance_

    #Obtain 3 top-loading attributes
    #My data has 4 intrinsic dimensions
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(x_str)
    loading_df=pca.components_.T * np.sqrt(pca.explained_variance_)
    attributeName=pd.DataFrame()
    attributeName['VariableName']=strati_df2.columns.values
    significanceValues=pd.DataFrame(data=loading_df,columns=['PC1','PC2','PC3','PC4'])
    significance_df=pd.concat([attributeName,significanceValues],sort=True,axis=1)
    significance = significance_df.drop(['VariableName'],axis=1)
    significance_df['SumOfSquaredLoadings']=significance\
    .apply(lambda x:np.sqrt(np.square(x['PC1'])+np.square(x['PC2'])+np.square(x['PC3'])+np.square(x['PC4'])),axis=1)
    s_df_sort = significance_df.sort_values(by=['SumOfSquaredLoadings'],ascending=False)


    app.run(debug=True)
