<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# # Analyze Taylor Swift Lyrics with Python
# 
# ### Try some of these resources for extra help as you work:
# 
# * [View the Analyze Taylor Swift Lyrics with Python Cheatsheet](https://www.codecademy.com/learn/case-study-analyze-taylor-swift-lyrics/modules/analyze-taylor-swift-lyrics/cheatsheet)
# * Learn more about analyzing text in [Apply Natural Language Processing with Python](https://www.codecademy.com/learn/paths/natural-language-processing)
# 
# [View the solution notebook](./solution.html)
# 
# [The original dataset was curated by Jan Llenzl Dagohoy and published on Kaggle](https://www.kaggle.com/datasets/thespacefreak/taylor-swift-song-lyrics-all-albums)
# 

# In[1]:


get_ipython().run_line_magic(&#39;matplotlib&#39;, &#39;inline&#39;)

import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter


# ### 1. Load the dataset
# 
# After running the first cell to load all necessary libraries, we need to load our dataset. 
# * Using pandas, load the dataset taylor_swift_lyrics_2006-2020_all.csv and save it as lyrics. 
# * Inspect the first few rows.
# * Use `.info()` to determine how many rows there are, check for missing values, and check the variable types.
# 

# *Note that we added the lyrics from the new Midnights album. If you want to inspect the lyrics from that album, you can find them in the file tree. Click on the Jupyter logo in the upper left corner and you can see all the files.* 
# 
# *Use &#34;taylor_swift_lyrics_2006-2022_all.csv&#34; in the `pd.read_csv()` function below to also analyze Midnights.*

# In[2]:


# load dataset
lyrics = pd.read_csv(&#34;taylor_swift_lyrics_2006-2020_all.csv&#34;)

#inspect the first few rows
## YOUR CODE HERE ##
lyrics.head()


# In[3]:


#get info about the DataFrame
## YOUR CODE HERE ##
lyrics.info()


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
# The Taylor Swift lyrics dataset consists of comprehensive data on the lyrics from her albums 2020. There are no missing values. We can see the lyric details along the column axis and each lyric along the row axis. We can find the name  of each album, the track, and the line in which the lyric appears. We do not have year associated with this dataset. 
# 
# </details>

# ### 2. Add essential data
# 
# We know that we want to explore her use of terms over years, but this dataset doesn&#39;t have years. We can either merge a dataset or manually create a new column. We have created a function for you that fills in the year based on the album name. 
# 
# * Apply this function to the `lyrics` DataFrame to create a new column.
# * Inspect the first few rows of the DataFrame

# In[4]:


# get a list of all the albums in this collection
## YOUR CODE HERE ##
album_list = lyrics[&#39;album_name&#39;].unique()
print(album_list)


# In[5]:


# this is a function to map the name of the album to the year it was released
def album_release(row):  
    if row[&#39;album_name&#39;] == &#39;Taylor Swift&#39;:
        return &#39;2006&#39;
    elif row[&#39;album_name&#39;] == &#39;Fearless (Taylor’s Version)&#39;:
        return &#39;2008&#39;
    elif row[&#39;album_name&#39;] == &#39;Speak Now (Deluxe)&#39;:
        return &#39;2010&#39;
    elif row[&#39;album_name&#39;] == &#39;Red (Deluxe Edition)&#39;:
        return &#39;2012&#39;
    elif row[&#39;album_name&#39;] == &#39;1989 (Deluxe)&#39;:
        return &#39;2014&#39;
    elif row[&#39;album_name&#39;] == &#39;reputation&#39;:
        return &#39;2017&#39;
    elif row[&#39;album_name&#39;] == &#39;Lover&#39;:
        return &#39;2019&#39;
    elif row[&#39;album_name&#39;] == &#39;evermore (deluxe version)&#39;:
        return &#39;2020&#39;
    #ok, we know folklore was actually released in Dec 2020, but this will make our analysis easier
    elif row[&#39;album_name&#39;] == &#39;folklore (deluxe version)&#39;:
        return &#39;2021&#39;
    #this is slightly differently formatted because the album name is recorded two ways.
    elif &#39;midnights&#39; in row[&#39;album_name&#39;]:
        return &#39;2022&#39;
    
    return &#39;No Date&#39;


# apply the function to the album
## YOUR CODE HERE ##
lyrics[&#39;album_year&#39;] = lyrics.apply(album_release, axis=1)
# inspect the first few rows of the DataFrame
## YOUR CODE HERE ##
lyrics.head()


# ### 3. Clean the lyric text
# 
# To accurately count keyword mentions, we need to make everything lowercase everything, remove punctuation, and exclude stop words. 
# 
# * Change everything to lower case and save the result in a new column called `clean_lyric`.
# * Remove punctuation and save the result to the existing `clean_lyric` column.
# * Run the code we provided to remove stopwords.
# * Check to be sure you have what you expect by viewing the first few rows.

# In[6]:


#lowercase
## YOUR CODE HERE ##
lyrics[&#39;clean_lyric&#39;] = lyrics[&#39;lyric&#39;].str.lower()

#remove punctuation
## YOUR CODE HERE ##
lyrics[&#39;clean_lyric&#39;] = lyrics[&#39;clean_lyric&#39;].apply(lambda x: &#39;&#39;.join([char for char in x if char not in string.punctuation]))
lyrics.insert(4, &#39;clean_lyric&#39;, lyrics.pop(&#39;clean_lyric&#39;))
lyrics.head()                                                


# In[7]:


#remove stopwords (see the next cell for illustration)
#create a small list of English stop words, feel free to edit this list
stop = [&#39;the&#39;, &#39;a&#39;, &#39;this&#39;, &#39;that&#39;, &#39;to&#39;, &#39;is&#39;, &#39;am&#39;, &#39;was&#39;, &#39;were&#39;, &#39;be&#39;, &#39;being&#39;, &#39;been&#39;]


#there are three steps in one here - explained below
#we make a list of words with `.split()`
#then we remove all the words in our list
#then we join the words back together into a string
lyrics[&#39;clean_lyric&#39;] = lyrics[&#39;clean_lyric&#39;].apply(lambda x: &#39; &#39;.join([word for word in x.split() if word not in (stop)]))


# In[8]:


# inspect the first few rows of the DataFrame
## YOUR CODE HERE ##
lyrics.head()


# ### The next 4 cells are for illustration only

# In[ ]:


#see what `.split()` does
lyrics[&#39;clean_lyric_list&#39;] = lyrics[&#39;clean_lyric&#39;].apply(lambda x: x.split())
print(lyrics.head())


# In[ ]:


#see what `.join()` does
lyrics[&#39;clean_lyric_list_rejoined&#39;] = lyrics[&#39;clean_lyric_list&#39;].apply(lambda x: &#39; &#39;.join(x))
print(lyrics.head())


# In[ ]:


#remove those illustration columns
lyrics.drop([&#39;clean_lyric_list&#39;, &#39;clean_lyric_list_rejoined&#39;], axis=1, inplace=True)
print(lyrics.head())


# In[ ]:


#there are many pre-built lists of stopwords, including one from sklearn.
#Most exclude too many words to be appropriate for song lyric analysis.
from sklearn.feature_extraction import text
skl_stop = text.ENGLISH_STOP_WORDS
print(skl_stop)


# ### 4. Find keyword mentions
# 
# Because we are most interested in &#34;midnight&#34;, we will first count how many times midnight occurs in Taylor Swift&#39;s lyrics. 
# 
# * Create a new column to indicate if a lyric has &#34;midnight&#34; in it.
# * Check how many times midnight occurrs

# In[9]:


#create a new column to reflect if the lyrics contain midnight
## YOUR CODE HERE ##
lyrics[&#39;midnight_appearance&#39;] = lyrics[&#39;lyric&#39;].str.contains(&#39;midnight&#39;, case=False)
num_midnight = lyrics[&#39;midnight_appearance&#39;].sum()
print(num_midnight)
lyrics.head()


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
# Midnight is not very common -- it only appeared 9 times! 
# 
# </details>

# ### 5. Expand the keyword list
# 
# Midnight might not be the only way that Taylor Swift has talked about night. We need to expand our list. We&#39;ve made a list of night words and day words. Feel free to add or remove words to make the list your own. 
# 
# * Join the lists into a regular expression string using the `.join()` function and the `|` to indicate &#34;or&#34;
# * Create a new column for each word category (day, night, time) that evaluates the clean lyrics for the presence of the words in your regular expression.
# * Count how many times the words appeared and print the result to the screen
# * Inspect the first few rows of the lyrics DataFrame to be sure it looks as you expect

# In[10]:


#night, day, and other time-related words
night = [&#39;night&#39;,&#39;midnight&#39;, &#39;dawn&#39;, &#39;dusk&#39;, &#39;evening&#39;, &#39;late&#39;, &#39;dark&#39;, &#39;1am&#39;, &#39;2am&#39;, &#39;3am&#39;, &#39;4am&#39;]
day = [&#39;day&#39;, &#39;morning&#39;, &#39;light&#39;, &#39;sun&#39;, &#39;dawn&#39;, &#39;noon&#39;, &#39;golden&#39;, &#39;bright&#39;]
time = [&#39;today&#39;, &#39;tomorrow&#39;, &#39;yesterday&#39;]


# In[11]:


#create a regular expression string for each list of words
## YOUR CODE HERE ##
night_joined = &#39;|&#39;.join([word for word in night])
day_joined = &#39;|&#39;.join([word for word in day])
time_joined = &#39;|&#39;.join([word for word in time])
print(night_joined)
#create a new column for each category of words
## YOUR CODE HERE ##
lyrics[&#39;night_joined_appearance&#39;] = lyrics[&#39;clean_lyric&#39;].str.contains(night_joined, case=False)
lyrics[&#39;day_joined_appearance&#39;] = lyrics[&#39;clean_lyric&#39;].str.contains(day_joined, case=False)
lyrics[&#39;time_joined_appearance&#39;] = lyrics[&#39;clean_lyric&#39;].str.contains(time_joined, case=False)
#count the number of times each category of word appears in the lyrics
## YOUR CODE HERE ##
num_night_appearance = lyrics[&#39;night_joined_appearance&#39;].sum()
num_day_appearance = lyrics[&#39;day_joined_appearance&#39;].sum()
num_time_appearance = lyrics[&#39;time_joined_appearance&#39;].sum()
#print the count of each word category
## YOUR CODE HERE ##
print(num_night_appearance)
print(num_day_appearance)
print(num_time_appearance)


# In[12]:


#inspect the first few rows
## YOUR CODE HERE ##
lyrics.head()


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
# Taylor Swift has referenced time alot, there&#39;s enough mentions for us to potentially spot some trends. Also, she has mentioned day far more than night. This might be because of the search terms, so we might consider revisiting the list. Or it could be because she is actually more focused on day and light than she is on night and dark. 
# 
# </details>

# # Task Group 2

# ### 6. Visualize how Taylor Swift&#39;s mentions of time have changed over time.
# 
# Now that we know that she has referenced time, we can see if that has changed, and if she has been dropping any hints about her upcoming album. 
# * Create a new dataframe called `yearly_mentions` that groups her mentions by year, and takes the sum of the other columns. Reset the index on this DataFrame to make it easier to use with matplotlib
# * Create a line chart in matplotlib showing the night mentions over time.

# In[13]:


#create a new dataframe for yearly mentions that groups mentions by year
## YOUR CODE HERE ##
yearly_mentions = lyrics.groupby(&#39;album_year&#39;)[&#39;night_joined_appearance&#39;, &#39;day_joined_appearance&#39;, &#39;time_joined_appearance&#39;].sum().reset_index()
yearly_mentions.head()


# In[14]:


#plot the mentions of night over years
## YOUR CODE HERE ##
plt.figure(figsize=(10, 6))
plt.plot(yearly_mentions[&#39;album_year&#39;], yearly_mentions[&#39;night_joined_appearance&#39;], marker=&#39;o&#39;, color=&#39;blue&#39;, label=&#39;Night Mentions&#39;) #marker for symbol at each point, label for legend 
plt.title(&#39;Taylor Swift Night Mentions Over Time&#39;)
plt.xlabel(&#39;Album_Year&#39;)
plt.ylabel(&#39;Number of Mentions&#39;)
plt.legend()
plt.grid(True)
plt.show()


# 
# 

# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
# It looks like there was a peak in her mentioning night around 2012. It seems like those mentions have become less prevalent over time, with an actual dip in 2019! We had expected midnight to appear more often around then, based on rumors that she plans everything 3 years out.
# 
# </details>

# ### 7. What albums are the most night/day focused?
# Now that we have a table of counts, it&#39;s very easy to figure out which albums have the most mentions of day or night. But our DataFrame only has the years, not the album names. We need to add those back in.
# * Run the code that reinstates the album name for this dataframe.
# * Use `.sort_values()` to order the yearly_mentions table and find which year has the most mentions of night.
# * Sort the day column as well.

# In[15]:


#reinstate the album name
#read the album_year_name.csv
year_name = pd.read_csv(&#39;album_year_name.csv&#39;)

#sort both dataframes by year
yearly_mentions.sort_values(by=&#39;album_year&#39;, ascending=True, inplace=True)
year_name.sort_values(by=&#39;album_year&#39;, ascending=True, inplace=True)

#add the new column for album name
yearly_mentions[&#39;album_name&#39;] = year_name[&#39;album_name&#39;]


# In[16]:


#sort the lyrics by the night column to find the albums with the most night references
## YOUR CODE HERE ##
night_most_mentions = yearly_mentions.sort_values(&#39;night_joined_appearance&#39;, ascending=False)
print(night_most_mentions)


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
# Her most &#34;nightly&#34; album was reputation in 2012, whereas her least nightly album was also the one right before Midnight. Has she been saving all her night-related lyrics for the past 3 years?
# 
# </details>

# In[17]:


#sort the lyrics by the day column to find the albums with the most day references
## YOUR CODE HERE ##
day_most_mentions = yearly_mentions.sort_values(&#39;day_joined_appearance&#39;, ascending=False)
print(day_most_mentions)


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
# It seems like her older work focused on day much more than her more recent. There was a period of time (2008-2012) when she was really focused on day, but it seems to have become less central to her work more recently. 
# 
# </details>

# ### 8. Compare day to night mentions
# * Create another line chart with both the night mentions and the day mentions

# In[18]:


#create a plot with one line showing number of night references by year 
#and another line with the number of day references by year
## YOUR CODE HERE ##
plt.figure(figsize=(10, 6))
plt.plot(yearly_mentions[&#39;album_year&#39;], yearly_mentions[&#39;night_joined_appearance&#39;], marker=&#39;o&#39;, label=&#39;Night Mentions&#39;, color=&#39;blue&#39;)
plt.plot(yearly_mentions[&#39;album_year&#39;], yearly_mentions[&#39;day_joined_appearance&#39;], marker=&#39;o&#39;, label=&#39;Day Mentions&#39;, color=&#39;orange&#39;)

# Adding labels and title
plt.xlabel(&#39;Album_Year&#39;)
plt.ylabel(&#39;Number of Mentions&#39;)
plt.title(&#39;Comparison of Day and Night Mentions Over Time&#39;)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
# Well this is interesting. We see a similar dip in referencing time around 2017, but a sharp increase in both 2019 and 2020 to reference day. 
# 
# </details>

# ### 9. Investigate position of day vs. night mentions within albums
# Maybe her references to time aren&#39;t always about the year that she released the album, but the ebb and flow of the album itself. Let&#39;s plot where, within each album, references to day and night occur.
# 
# * Create a variable that indicates the position of a lyric within an album. Call this &#39;position&#39;.
# * Create a new dataframe called positional_mentions that groups her mentions by album position, and takes the sum of the other columns. Reset the index on this DataFrame to make it easier to use with matplotlib.
# * Create a line chart in matplotlib showing the night and day mentions as a position within the album.
# 
# *Remember that you can always print the head of the DataFrame to check your work*

# In[19]:


#create a position variable that includes both the track number and line number
## YOUR CODE HERE ##
lyrics[&#39;position&#39;] = lyrics.groupby(&#39;album_name&#39;).cumcount() + 1 #DONT KNOW IF THIS STUFF IS RIGHT DONT THINK IT MATTERS
#create a new DataFrame that is grouped by position
## YOUR CODE HERE ##
positional_mentions = lyrics.groupby([&#39;position&#39;]).agg({
    &#39;night_joined_appearance&#39;: &#39;sum&#39;,
    &#39;day_joined_appearance&#39;: &#39;sum&#39;
}).reset_index()
positional_mentions.head()


# In[20]:


#increase the size of the plot 
fig = plt.gcf()
fig.set_size_inches(25,10)

#create a plot with two lines to show frequency of day vs. night references by position in the album
## YOUR CODE HERE ##
plt.figure(figsize=(10, 6))
plt.plot(positional_mentions[&#39;position&#39;], positional_mentions[&#39;night_joined_appearance&#39;], marker=&#39;o&#39;, label=&#39;Night Mentions&#39;, color=&#39;blue&#39;)
plt.plot(positional_mentions[&#39;position&#39;], positional_mentions[&#39;day_joined_appearance&#39;], marker=&#39;o&#39;, label=&#39;Day Mentions&#39;, color=&#39;orange&#39;)

# Adding labels and title
plt.xlabel(&#39;Position Within Album&#39;)
plt.ylabel(&#39;Number of Mentions&#39;)
plt.title(&#39;Night and Day Mentions Across Album Positions&#39;)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
#     Honestly not that much -- we can&#39;t see any strong trends here. Best to move on. 
# </details>

# # Task Group 3

# ### 10. Tokenize the Lyrics
# It&#39;s great to know how much she has focused on day and night, but we would also like to do a little more sophisticated analysis. Before we can work with our cleaned lyrics, we will have to tokenize them. Tokenization is a special way of breaking up words that is a little more nuanced than just using white space. The output is a list of words that we can then perform text analysis on.
# 
# We will use the `word_tokenize` function from NLTK (the Natural Language ToolKit), and apply it to every row in our DataFrame with a lambda expression.
# 
# * Run the cell to tokenize the cleaned lyrics.
# * Inspect the first few rows of the lyrics DataFrame
# * Create a list of all the tokens in the lyrics_tok column into one list
# * Use the Counter function from the collections package to count the number of times each word appears
# * Sort the resulting dictionary
# 

# In[21]:


#run this cell to tokenize the words in the clean_lyric column
lyrics[&#39;lyrics_tok&#39;] = lyrics[&#39;clean_lyric&#39;].str.split(&#39; &#39;)


# In[22]:


#inspect the first few lines
## YOUR CODE HERE ##
lyrics.head()


# In[23]:


#determine what words overall are the most frequently used words
#create a list of all the words in the lyrics_tok column
word_list = [word for list_ in lyrics[&#39;lyrics_tok&#39;] for word in list_]
#print(word_list) dont need but its cool

#use the counter function to count the number of times each word appears
## YOUR CODE HERE ##
word_counts = Counter(word_list)
#print(word_counts)
#sort the word frequencies to find out the most common words she&#39;s used. 
## YOUR CODE HERE ##
sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
#call the word frequency
## YOUR CODE HERE ##
#no need sorted calls it


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
#    She mostly talks about you and then herself, while that&#39;s not surprising, it also tells us that she&#39;s mostly writing about relationships and her realtionships with others. 
# </details>

# ### 11. Analyze Lyric Sentiment
# 
# To better understand how she thinks and feels about day and night, we want to know which one she writes about more favorably. 
# 
# We are going to use the pre-trained sentiment classifier that comes with NLTK. It&#39;s it trained on tweets and therefore best for short text. It return 4 values: positive, negative, neutral, and compound. Compound will be of special interest to us. It is the normalized sum of positive and negative. 
# 
# * Run the sample sentiment analyzer cell to see how the SentimentIntensityAnalyzer works.
# * Apply the sia sentiment analyzer to the clean_lyric column of the DataFrame with a lambda expression
# * Run the cell that transforms the dictionary into columns of the DataFrame

# In[24]:


#run this cell to add a package from NLTK for our sentiment analyzer.
nltk.download(&#39;vader_lexicon&#39;)


# In[25]:


#run this cell to see how the sentiment analyzer works
sia = SentimentIntensityAnalyzer()
sia.polarity_scores(&#34;I love Taylor Swift!&#34;)


# In[26]:


#create a new column called polarity and apply the sia method to the clean_lyric column with a lambda expression
## YOUR CODE HERE ##
lyrics[&#39;polarity&#39;] = lyrics[&#39;clean_lyric&#39;].apply(lambda x: sia.polarity_scores(x))


# In[27]:


#run this cell to transform the polarity dictionary into columns of the DataFrame
lyrics[[&#39;neg&#39;, &#39;neu&#39;, &#39;pos&#39;, &#39;compound&#39;]] = lyrics[&#39;polarity&#39;].apply(pd.Series)
lyrics.drop(&#39;polarity&#39;, axis=1)


# In[28]:


#inspect the first few rows
## YOUR CODE HERE ##
lyrics.head()


# ### 12. Corpus Sentiment Analysis
# 
# Now that we have sentiment for all of her lyrics, let&#39;s learn more about her overall sentiment, as well as how that has change throughout her career. 
# * Use the pos, neg, and compound columns to calculate the overall sentiment of her entire collection, and print the result.
# * Use the groupby function and matplotlib to visualize the overall sentiment of her albums over time 

# In[29]:


#calculate overall sentiment for pos, neg, sentiment
## YOUR CODE HERE ##
overall_pos = sum(lyrics[&#39;pos&#39;])
overall_neg = sum(lyrics[&#39;neg&#39;])
overall_compound = sum(lyrics[&#39;compound&#39;])
#print the overall sentiments
## YOUR CODE HERE ##
print(overall_pos)
print(overall_neg)
print(overall_compound)


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
#    Overall Taylor Swift&#39;s lyrics are mostly positive! 
# </details>

# In[32]:


#create a new DataFrame using the groupby method for the album_year
## YOUR CODE HERE ##
grouped_albums = lyrics.groupby(&#39;album_name&#39;)
#visualize sentiment over time 
## YOUR CODE HERE ##
total_sentiment = grouped_albums[[&#39;pos&#39;, &#39;neg&#39;, &#39;compound&#39;]].sum()
total_sentiment.plot(kind=&#39;bar&#39;, stacked=True)
plt.title(&#39;Overall Sentiment of Albums Over Time&#39;)
plt.xlabel(&#39;Album&#39;)
plt.ylabel(&#39;Total Sentiment Score&#39;)
plt.legend([&#39;Positive&#39;, &#39;Negative&#39;, &#39;Compound&#39;])
plt.show()


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
#    Her degree of positivity has really fluctuated over time. Her 2020 album was by far the least positive, though all of her albums are actually positive. 
# </details>

# ### 13. Day or Night? Positive or Negative
# Finally the big reveal. Does Taylor Swift write more positively about day or night? 
# * Create two DataFrames: one for all the lyrics that mention night and one for all the lyrics that mention day.
# * Check that each DataFrame has the right number of values with the `len()` function
# * Calculate the sentiment for both night and day using the compound value and print them.

# In[38]:


#create a DataFrame filtered for only night mentions
## YOUR CODE HERE ##
night_only = lyrics[lyrics[&#39;night_joined_appearance&#39;]==True] #get dataframes with sentiement of only night lyrics
#create a DataFrame filtered for only day mentions
## YOUR CODE HERE ##
day_only = lyrics[lyrics[&#39;day_joined_appearance&#39;]==True] #get dataframes with sentiement of only day lyrics
#print the length of the night and day DataFrames
## YOUR CODE HERE ##
print(len(night_only)) #should be 240
print(len(day_only)) #should be 363


# In[39]:


#calculate the sentiment of each day and night DataFrame from the compound values
## YOUR CODE HERE ##
total_night_sentiment = night_only[&#39;compound&#39;].sum()
total_day_sentiment = day_only[&#39;compound&#39;].sum()
#print the results
## YOUR CODE HERE ##
print(total_night_sentiment) #should be 17.2783
print(total_day_sentiment)  #should be 25.9887


# <details>
#     <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>
# 
# She&#39;s still overall more positive in her sentiment about day than night! 
#     
# You have the tools to explore if that sentiment has changed over time, or if her feelings towards night have grown more positive or negative over time. And of course, we would love to see your analysis of her new album, Midnight. Have her feelings changed?
#     
# Share your findings in the Codecademy forums! 
# </details>

# In[ ]:




<script type="text/javascript" src="https://www.codecademy.com/assets/relay.js"></script></body></html>