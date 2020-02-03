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


#Here is the main fucntion
if __name__ == "__main__":
    df = pd.read_csv('StateHousingPrice.csv')

    #devide the midean housing price into 5 groups according to the price level
    def function_label(a):
        if a < 150000:
            return 'very low-price(<150k)'
        if a>=150000 and a< 250000:
            return 'low-price(150-250k)'
        if a>=250000 and a< 350000:
            return 'medium-price(250-350k)'
        if a>=350000 and a< 450000:
            return 'high-price(350-450k)'
        else:
            return 'very high-price(450k+)'

    #Creating dataframe for each State, for (1)precise data clean, (2)major attribute analysis
    #Obtain each State's data by time sequence by calling the function below:
    def function(s):
        State_df = pd.DataFrame()
        State_df = df[df['RegionName'] == s]
        State_df = State_df.fillna(State_df.mean().apply(np.round))
        #add the target column to the stritified data frame
        State_df['price'] = State_df.apply(lambda x: function_label(x.MedianListingPrice_AllHomes), axis = 1)
        target = State_df.loc[:,['price']].values
        State_df.to_csv(s,sep=',')
        return State_df

    #This function is to obtain 3 top-loading attributes of each State
    #Keep 4 intrinsic dimensions
    def function_HAttri(state):
        global top3_housing_df
        df = pd.read_csv(state)
        pca = PCA(n_components=4)
        df2 = df.drop(['Unnamed: 0','Date','RegionName','MedianListingPricePerSqft_AllHomes',\
                       'MedianListingPrice_AllHomes','ZHVI_AllHomes','price','ZHVI_TopTier',\
                       'ZHVI_MiddleTier','ZHVI_BottomTier','Sale_Prices'],axis=1)
        df2 = df2.dropna(axis=1, how='all')
        df3 = StandardScaler().fit_transform(df2)
        prinComps = pca.fit_transform(df3)
        loading_df = pca.components_.T * np.sqrt(pca.explained_variance_)
        attributeName=pd.DataFrame()
        attributeName['VariableName']=df2.columns.values
        significanceValues=pd.DataFrame(data=loading_df,columns=['PC1','PC2','PC3','PC4'])
        significance_df=pd.concat([attributeName,significanceValues],sort=True,axis=1)
        significance = significance_df.drop(['VariableName'],axis=1)
        significance_df['SumOfSquaredLoadings']=significance\
        .apply(lambda x:np.sqrt(np.square(x['PC1'])+np.square(x['PC2'])+np.square(x['PC3'])+np.square(x['PC4'])),axis=1)

        #sort the attributes by high-low loaded significance order
        sort_df = significance_df.sort_values(by=['SumOfSquaredLoadings'],ascending=False)

        #fetch the top3 dominant house styles of this state
        top3_df = pd.DataFrame(columns = ['State','Dominant1','Dominant2'])
        top3_df.loc[0,'State'] = state
        top3_df.loc[0,'Dominant1'] = sort_df.iloc[0,0]
        top3_df.loc[0,'Dominant2'] = sort_df.iloc[1,0]
        top3_housing_df = pd.concat([top3_housing_df, top3_df],axis = 0)
        return top3_housing_df

    def pca_plot(state):
        df = pd.read_csv(state)
        df2 = df.drop(['Unnamed: 0','Date','RegionName','MedianListingPricePerSqft_AllHomes',\
                       'MedianListingPrice_AllHomes','ZHVI_AllHomes','price'],axis=1)
        df2 = df2.dropna(axis=1, how='all')
        pca = PCA(n_components=2)
        df3 = StandardScaler().fit_transform(df2)
        principalComponents = pca.fit_transform(df3)
        principal2_df = pd.DataFrame(data=principalComponents,columns=['PC1','PC2'])
        principal2_df['price'] = df['price']
        data = principal2_df
        data.to_csv(state+'_pca',sep=',')
        return data

    #load US population csv file
    pop_df = pd.read_csv('2010_2017_agesex-civ.csv')
    #divide age into groups
    def pop_divide(a):
        if a>= 0 and a < 19:
            return '0-18'
        if a>=19 and a< 31:
            return '19-30'
        if a>=31 and a< 46:
            return '31-45'
        if a>=46 and a< 65:
            return '46-64'
        if a>=65 and a< 85:
            return '65+'

    #divide each State's population into different age groups
    def state_pop(s):
        global pop_df
        State_pop_df = pop_df[pop_df['NAME'] == s]
        State_pop_df['AgeRange'] = State_pop_df.apply(lambda x: pop_divide(x.AGE), axis = 1)
        AgeRange = State_pop_df.loc[:,['AgeRange']].values
        State_pop_df.to_csv(s.replace(' ', '') + '_population',sep=',')
        return State_pop_df

    #calculate each State's population by year from 2010 to 2017
    def state_pop1(state):
        s = state + '_population'
        df = pd.read_csv(s)
        df = df.drop(['Unnamed: 0','SUMLEV','REGION','DIVISION','STATE',\
                          'NAME','SEX','AGE','ESTBASE2010_CIV'],axis=1)
        range_df = df.groupby(by=['AgeRange']).sum().T
        return range_df

    #calculate correlations between 1-5 bedroom and condocoop styles and
    #different agerange groups
    def corr(s):
        house_df = pd.read_csv(s)
        house_df = house_df.fillna(house_df.mean().apply(np.round))
        house_df = house_df[['Date','RegionName','MedianListingPrice_1Bedroom','MedianListingPrice_2Bedroom',\
                             'MedianListingPrice_3Bedroom','MedianListingPrice_4Bedroom',\
                             'MedianListingPrice_5BedroomOrMore']]
        house_df.columns = ['Date','RegionName','1-B','2-B','3-B','4-B','5-B']
        house_df = house_df.dropna(axis=1, how='all')
        for i in range(0,len(house_df['Date'])):
            house_df['Date'][i] = house_df['Date'][i][:4]
        house_df=house_df.groupby(by=['Date']).mean()
        df2 = state_pop1(s)
        df2.index=['2010','2011','2012','2013','2014','2015','2016','2017']
        house_df.index=['2010','2011','2012','2013','2014','2015','2016','2017']
        house_df = house_df.join(df2)
        corr = house_df.corr(method='pearson')
        corr.to_csv(s + '_corrMatrix',sep=',')
        print(corr)

    #fecth State names in a list
    stateNames_df = df['RegionName'].unique()

    #create each State's housing price csv file
    for s in stateNames_df:
        if s == 'UnitedStates':
            pass
        else:
            function(s)

    #Create 50 States' top3 attributes significance table in one csv file
    top3_housing_df = pd.DataFrame(columns = ['State','Dominant1','Dominant2'])
    for s in stateNames_df:
        if s == 'UnitedStates':
            pass
        else:
            function_HAttri(s)
    top3_housing_df.to_csv('dominsInOne.csv',sep=',')

    #create each State's 2-D PCA plot csv file
    for s in stateNames_df:
        if s == 'UnitedStates' or s == 'NorthDakota':
            pass
        else:
            pca_plot(s)

    stateNames_df2 = pop_df['NAME'].unique()
    #create each State's population statistics csv file
    for s in stateNames_df2:
        if s == 'United States':
            pass
        else:
            state_pop(s)

    #create correlation matrix for each State
    for s in stateNames_df:
        if s == 'UnitedStates' or s == 'NorthDakota':
            pass
        else:
            corr(s)

