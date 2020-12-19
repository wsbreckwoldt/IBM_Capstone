#!/usr/bin/env python
# coding: utf-8

# Data Wrangling and Cleaning 

# In[1]:


#import libraries 
import requests #for performing HTTP requests
import urllib.request
from bs4 import BeautifulSoup #for wrangling HTML content 
import numpy as np
import pandas as pd
from urllib.request import urlopen


# In[2]:


df = pd.read_html("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")
#read the wikipedia page, returns a list of dataframes.


# In[3]:


len(df)
#list has length of 3


# In[4]:


df[0].to_csv("postal_codes_data.csv")
#first df on the list is what we are looking for now must clean


# In[5]:


postal_codes_data=pd.read_csv("postal_codes_data.csv")
#changed the dataframe in the list to a .csv file


# In[6]:


postal_codes_data.drop(["Unnamed: 0"], axis=1, inplace=True) 
#removed the unnecessary column


# In[7]:


postal_codes_data.head()


# In[8]:


#i removed Postal Codes without Boroughs
postal_codes_data = postal_codes_data[postal_codes_data.Borough !="Not assigned"]


# In[9]:


postal_codes_data.head()
#"Not assigned" values have been removed from column("Borough")


# In[10]:


postal_codes_data["Neighbourhood"].replace("Not assigned", "Borough")


# In[11]:


postal_codes_data.shape


# Getting the Latitude and Longtitude of the Postal Codes (Part 2)

# In[12]:


geo_df=pd.read_csv('http://cocl.us/Geospatial_data')
#had issues with geocoder, so read csv with lat and long


# In[13]:


geo_df.head()
#display the Geospatal data


# In[14]:


geo_merged=pd.merge(geo_df, postal_codes_data, on='Postal Code')
#merging the two dataframes


# In[15]:


geo_data=geo_merged[['Postal Code', 'Borough', 'Neighbourhood', 'Latitude','Longitude']]
#updating the dataframe so that the columns in the correct order


# In[19]:


geo_data.head()
#the new dataframe


# In[18]:


geo_data.rename(columns={'Postal Code' : 'PostalCode', 'Neighbourhood' : 'Neighborhood'}, inplace=True)


# Exploring and Clustering the Neighborhoods in Toronto (Part 3)

# In[19]:


toronto_data=geo_data[geo_data['Borough'].str.contains("Toronto")]
toronto_data.head()


# In[21]:


CLIENT_ID = 'C0SHEUS3TZNS53RVI1WEQQMQGJUIJ1CVPYSTA54RVP1T3LXQ'

CLIENT_SECRET = '1KRDEUYMOSEZX2FLXM1TLTJ3HAJG40U1XQAOCG0UGDXHKRUX'

VERSION = '20180605'


# In[22]:


def getNearbyVenues(names, latitudes, longitudes):
    radius=500
    LIMIT=100
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[25]:


toronto_venues = getNearbyVenues(names=toronto_data['Neighborhood'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )


# In[26]:


toronto_venues.head()


# In[28]:


toronto_venues.groupby('Neighborhood').count()


# In[29]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
toronto_onehot.drop(['Neighborhood'],axis=1,inplace=True) 
toronto_onehot.insert(loc=0, column='Neighborhood', value=toronto_venues['Neighborhood'] )
toronto_onehot.shape


# In[31]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped.head()


# In[32]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[33]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[35]:


from sklearn.cluster import KMeans

# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[36]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head()


# In[37]:


neighborhoods_venues_sorted.head()


# In[48]:


get_ipython().system('pip install geopy')

from geopy.geocoders import Nominatim
address = 'Toronto, CA'


geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))


# Creating the Map

# In[53]:


import matplotlib.cm as cm
import matplotlib.colors as colors
import folium
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:




