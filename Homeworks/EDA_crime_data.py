#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ## Data Input

# All the necessary packages are loaded, data is input into the system and the first 10 rows are visualized. This is followed by checking the shape and datatypes of data present in each column. All these steps are done to familiarize the data in hand. 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#input data and check the first 10 rows

crimedf = pd.read_csv("Chicagocrimes.csv")
crimedf.head(10)


# In[5]:


crimedf.info()


# ## Data Cleaning and Pre-processing

# Data is checked for the presence of NA values. Treating the missing values is an essential step in all Data analysis process so as to avoid any further complications. It was found that the columns Case Number, Block, Ward, Community Area, FBI Code, Updated on, Latitude, Longtitude and Location were redundant and not necessary for the analysis. Thus, these are removed initially. In the rest of the dataset, the columns 'Location Description', 'X Coordinate' and 'Y Coordinate' had missing values which constituted less tthan 10% of the total data. Hence, it was decided to remove these rows too.

# In[6]:


#count of nas in each column

crimedf.isnull().sum(axis = 0)


# In[7]:


#drop unnecessary columns and drop nas

crimedf.drop(labels  = ['Case Number',
                        'Block',
                        'Ward',
                        'Community Area',
                        'FBI Code',
                        'Updated On', 
                        'Latitude', 
                        'Longitude', 
                        'Location'], inplace = True, axis = 1)
crimedf.dropna(inplace = True)


# Following this, the Date value given in the dataset was converted into the pandas recognizable DateTime format for the purpose of analysis and the values of Month, Date, Time, Hour and Day of Week are extracted from it.

# In[8]:


#covert datetime into pandas datetime format

crimedf.Date = pd.to_datetime(crimedf.Date, format = '%m/%d/%Y %I:%M:%S %p')
crimedf.index = pd.DatetimeIndex(crimedf.Date)


# In[9]:


crimedf['Month'] = crimedf.Date.dt.month
crimedf['date'] = crimedf.Date.dt.date
crimedf['DayofWeek'] = crimedf.Date.dt.weekday_name
crimedf['Time'] = crimedf.Date.dt.time
crimedf['Hour'] = crimedf.Date.dt.hour
crimedf.head()


# ## Exploratory Data Analysis
# 
# ### Insights on Crimes and Date/Times

# To get insights on crimes and the times at which it is happening, crimes are categorized by the primary type and sorted according to the values of occurences. A bar plot is used to visualize the variation.

# In[17]:


plt.figure(figsize=(8,10))
crimedf.groupby([crimedf['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Most occuring crimes by Primary Type')
plt.ylabel('Crime Type')
plt.xlabel('Count')
plt.show();


# The plot above shows Type of crimes versus their counts. Thefts are the most common crime in Chicago followed by Battery, criminal damage, narcotics and assault. 
# 
# Monthly variations of crimes are also checked to identify which months contribute to more crime.

# In[16]:


crimedf.groupby('Month').size().plot(kind = 'barh')
plt.ylabel('Month')
plt.xlabel('Count')
plt.title('Number of crimes per month')
plt.show();


# It can be observed that Summer months (mainly months of july and august) had more number of crimes and the numbers are relatively lesser during the months of december, january and february (winter months). This can be attributed to the very well known harsh Chicago Winters.

# The next visualized trend is the variation of crimes according to the day of the week. 

# In[15]:


crimedf.groupby('DayofWeek').size().plot(kind = 'barh')
plt.ylabel('Day of the week')
plt.xlabel('Count')
plt.title('Number of crimes by day of the week')
plt.show;


# It is observed that there is not much change in number of crimes with days of week. The values are almost the same with slightly high values for Friday (but no significant difference between other days).
# 
# Finally, number of crimes versus hour of day is compared. This is also visualized using bar plots. 

# In[14]:


crimedf.groupby('Hour').size().plot(kind = 'bar')
plt.ylabel('Hour of the day')
plt.xlabel('Count')
plt.title('Number of crimes by hour of the day')
plt.show;


# It can be clearly observed from the graph that hours of the noght contribute to more crime (mainly starting from the 18th hour to the 0th hour). The only exception that is seen is the peak during the 12th hour of the day. 
# 
# To get more clarity, the top 5 crimes for each hour of the day is found out and visualized as shown below. 

# In[18]:


t5hour = crimedf.groupby(['Hour', 'Primary Type']).size().reset_index(name='Counts').groupby('Hour').apply(lambda x: x.sort_values('Counts', ascending = False).head(5))
g =sns.catplot("Primary Type", y='Counts', col='Hour', col_wrap=4, data=t5hour, kind='bar')
for ax in g.axes:
    plt.setp(ax.get_xticklabels(), visible=True, rotation=30, ha='right')

plt.subplots_adjust(hspace=0.4);


# The above plot throws light on types of crime occuring at each hour of the day. We can observe that 'THEFT' peaks almost at every hour of the day. But if we take the case of the night hours especially from 20th hour to 3rd hour, we can observe peak in 'BATTERY' (exception being 0th hour). Also, 'THEFT' is found to be very high during the 12th hour of the day also.

# ### Insights on Crime versus Year

# To understand how number of crimes varies across the years, a line chart is plotted. 

# In[19]:


plt.figure(figsize=(15,5))
crimedf.resample('M').size().plot(legend=False)
plt.title('Number of crimes per month')
plt.xlabel('Months')
plt.ylabel('Count')
plt.show();


# It can be clearly seen that over the period of years, the number of crimes has decreased consderably. The decreased number of crimes in the latter years of 2017 - 2019 can also be due to the reduced number of datapoints too. But the general trend is decreasing. Also, number of crimes tend to peak during the middle of the year (mostly the summer months) and go down towards the end of each year.
# 
# A heatmap is also used to understand these variations more clearly

# In[20]:


crime_monthyr = pd.DataFrame(crimedf.groupby(['Month', 'Year']).size().sort_values(ascending = False).reset_index(name = 'Count'))
monthyearplot = crime_monthyr.pivot_table(values='Count',index='Month',columns='Year')

plt.figure(figsize = (8,5))
sns.heatmap(monthyearplot, cmap = 'YlGnBu');


# The heatmap shows highest values of crime for the month of August in the year 2002. The months between April to October for the years from 2001 - 2008 shows very high numbers of Crimes than the rest of the months and years, the least during the early months of January and February for the years 2017 - 2020.
# 
# Variations across types of crimes for the given years is visualized as given below:

# In[85]:


crimes_count_date = crimedf.pivot_table('ID', aggfunc=np.size, 
                                        columns='Primary Type', 
                                        index=crimedf.index.date, 
                                        fill_value=0)

crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)
pl = crimes_count_date.rolling(365).sum().plot(figsize=(20, 30), 
                                               subplots=True, 
                                               layout=(-1, 3), 
                                               sharex=False, 
                                               sharey=False)


# ### Insights on Arrests v/s no arrests

# In[21]:


crimetype = crimedf.groupby(['Primary Type', 'Arrest'])['Arrest'].size()
crimetype.unstack().plot(kind = 'bar',stacked = True,logy = True, figsize = (18,5))
plt.title('Arrest v/s No Arrest for types of Crime');


# The plot of arrest versus no arrests for different types of crimes indicates that very small number of crimes result in arrests. The Y-axis of the plot is the log(count) to get a more clear picture of how many crimes are converted into arrests. It can be observed that crimes like Burglary, criminal sexual assault, motor vehicle theft, robbery and stalking, hardly any arrests are made while concealed carry licensed violation, gambling, narcotics, prostitutionand public indecency results in a lot of arrests. 
# 
# The trend of arrests across the years is also checked. 

# In[22]:


crimetype = crimedf.pivot_table('ID', columns = 'Year', index = 'Arrest', aggfunc = np.size)
crimetypeplot = crimetype.T.plot(kind = 'bar', figsize = (20,5))
ylab = crimetypeplot.set_ylabel('Count')
plt.title('Arrest v/s No Arrest for Years');


# Across years, the number of arrests and no arrests decreases in the similar way. 

# ### Insights on Crime and Locations

# The next compared attributes are Types of Crimes and the location where they are committed. Firstly, a count of Crimes happening at different locations is taken. 

# In[23]:


plt.figure(figsize=(8,10))
crimedf.groupby([crimedf['Location Description']]).size().sort_values(ascending=True).tail(25).plot(kind='barh')
plt.title('Number of crimes by Location')
plt.ylabel('Crime Location')
plt.xlabel('Number of crimes')
plt.show();


# It can be observed that majority of crimes take place in streets followed by residence, apartments and sidewalks. 
# 
# The X coordinates and Y coordinates available in the data is utilized to roughly plot the map of Chicago according to each districts as shown below.

# In[24]:


crimedata = crimedf.loc[(crimedf['X Coordinate']!=0)]
sns.lmplot('X Coordinate', 
           'Y Coordinate',
           data=crimedata,
           fit_reg=False, 
           hue="District",
           palette='colorblind',
           height=5,
           scatter_kws={"marker": "D", 
                        "s": 10})
ax = plt.gca()
plt.figure(figsize = (10,25))
ax.set_title("A Rough map of Chicago\n", fontdict={'fontsize': 15}, weight="bold")
plt.show();


# Now, for each district in Chicago, we check the variations of top 3 crime types using bar plots. 

# In[25]:


distr = crimedf.groupby(['District', 'Primary Type']).size().reset_index(name='Counts').groupby('District').apply(lambda x: x.sort_values('Counts', ascending = False).head(3))
distr_g =sns.catplot("Primary Type", y='Counts', col='District', col_wrap=4, data=distr, kind='bar')
for ax in distr_g.axes:
    plt.setp(ax.get_xticklabels(), visible=True, rotation=30, ha='right')

plt.subplots_adjust(hspace=0.4);


# It was seen from the numbers that District 8 has the maximum number of crimes. From the multiple types of crimes, 6 types of crimes are chosen and visualized in the map with district wise demarcation. The chosen types are Theft (accounts for the most count), Battery (second highest), Narcotics (crimes with huge number of arrests), and randomnly chosen crimes like Homicide, Weapons Violation, Criminal Damage.

# In[26]:


col2 = ['Date','Primary Type','Arrest','Domestic','District','X Coordinate','Y Coordinate']
multiple_crimes = crimedf[col2]
multiple_crimes = multiple_crimes[multiple_crimes['Primary Type']                  .isin(['THEFT','HOMICIDE','BATTERY','NARCOTICS','WEAPONS VIOLATION','CRIMINAL DAMAGE'])]

multiple_crimes = multiple_crimes[multiple_crimes['X Coordinate']!=0]
g= sns.lmplot(x="X Coordinate", 
              y="Y Coordinate", 
              col="Primary Type",
              hue = 'District', 
              data=multiple_crimes,
              col_wrap=2, 
              height=6, 
              fit_reg=False, 
              sharey=False, 
              scatter_kws={"marker": "D","s": 10});


# ### Insights on Domestic and Arrests

# The ratio of arrests made for Domestic crimes versus non arrests is represented using a Pie chart as given below:

# In[29]:


crimedf[crimedf.Domestic]['Arrest'].value_counts(normalize=True).plot(kind='pie', legend=True, autopct='%2f')
plt.title('Percentage of Domestic Crimes with Arrest v/s No Arrests');


# ### Insights on subtypes of Thefts

# To identify how different types of Thefts varied within each other, a count of the multiple types were taken and visualized using a horizontal bar plot as shown below:

# In[30]:


plt.figure(figsize = (8,10))

crimedf[crimedf['Primary Type'] == 'THEFT']['Description'].value_counts().sort_values(ascending = True).plot(kind = 'barh')
plt.title('Types of thefts and their counts')
plt.xlabel('Count')
plt.ylabel('Description')
plt.show;


# It can be seen that Theft of $500 and under accounts to the maximum number in the subtype. 

# ### Insight on Types of Crimes and Location 

# To understand how types of crimes are distributed across different locations, first Primary type and location description are grouped and the variations are visualized usig heatmaps. 

# In[31]:


crime_monthyr = pd.DataFrame(crimedf.groupby(['Primary Type', 'Location Description']).size().sort_values(ascending = False).reset_index(name = 'Count')).head(20)
monthyearplot = crime_monthyr.pivot_table(values='Count',index='Primary Type',columns='Location Description')

plt.figure(figsize = (3,4))
sns.heatmap(monthyearplot, cmap = 'YlGnBu')
plt.title('Heatmap for crimes versus locations');


# The color for Street against Theft clearly indicates how high Street side thefts are. The heatmap is sparsely populated due to with hues higher for crimes like Criminal Damage, Motor vehhicle Theft and Narcotics mostly happening in the streets followed by residence or sidewalks. Its for the ease of reperesentation and understanding that only top 20 from the list was taken to be visualized in the heatmap.
