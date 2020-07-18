import requests
import pandas as pd

wiki = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
wikipedia_page = requests.get(wiki)

df_raw = pd.read_html(wikipedia_page.content, header=0)[0]
df_new = df_raw[df_raw.Borough != 'Not assigned']

df_new.head()

df_new.loc[df_new.Neighborhood == 'Not assigned']
df_new.Neighborhood.replace('Not assigned',df_new.Borough,inplace=True)
df_new.head(8)
df_toronto = df_new.groupby(['Postal Code', 'Borough'])['Neighborhood'].apply(lambda x: ', '.join(x))
df_toronto = df_toronto.reset_index()
df_toronto.rename(columns = {'Postal Code':'Post Code'}, inplace = True)
df_toronto.rename(columns = {'Neighborhood':'Neighbourhood'}, inplace = True)
df_toronto.head()
df_toronto.shape
url = 'http://cocl.us/Geospatial_data'
df_geo=pd.read_csv(url)
df_geo.head()