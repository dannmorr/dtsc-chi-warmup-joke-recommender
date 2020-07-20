# The Jester Dataset

![](https://vignette.wikia.nocookie.net/helmet-heroes/images/9/9b/Jester_Hat.png/revision/latest/scale-to-width-down/340?cb=20131023213944)

This morning we will be building a recommendation system using User ratings of jokes.

By the end of this notebook, we will know how to 
- Format data for user:user recommendation
- Find the cosign similarity between two vectors
- Use K Nearest Neighbor to indentify vector similarity
- Filter a dataframe to identify the highest rated joke based on K most similar users.


```python
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
```

### About user data
Format:

- Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
- One row per user
- The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
- The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. Almost all users have rated those jokes.



```python
df = pd.read_csv('./data/jesterfinal151cols.csv', header=None)
df = df.fillna(99)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>141</th>
      <th>142</th>
      <th>143</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
      <th>150</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>0.21875</td>
      <td>99</td>
      <td>-9.28125</td>
      <td>-9.28125</td>
      <td>99</td>
      <td>...</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-9.68750</td>
      <td>99</td>
      <td>9.93750</td>
      <td>9.53125</td>
      <td>99</td>
      <td>...</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-9.84375</td>
      <td>99</td>
      <td>-9.84375</td>
      <td>-7.21875</td>
      <td>99</td>
      <td>...</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>82</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>6.90625</td>
      <td>99</td>
      <td>4.75000</td>
      <td>-5.90625</td>
      <td>99</td>
      <td>...</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-0.03125</td>
      <td>99</td>
      <td>-9.09375</td>
      <td>-0.40625</td>
      <td>99</td>
      <td>...</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 151 columns</p>
</div>



### Joke data


```python
jokes = pd.read_table('./data/jester_items.tsv', header = None)
jokes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1:</td>
      <td>A man visits the doctor. The doctor says, "I h...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2:</td>
      <td>This couple had an excellent relationship goin...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3:</td>
      <td>Q. What's 200 feet long and has 4 teeth? A. Th...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4:</td>
      <td>Q. What's the difference between a man and a t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5:</td>
      <td>Q. What's O. J. Simpson's web address? A. Slas...</td>
    </tr>
  </tbody>
</table>
</div>



The 0 column is the join column we need to connect with the user dataframe. 

In the cell below, we 
- Remove the ':' character from the `0` column
- Convert the column to an integer datatype
- Set the `0` column as the index for our jokes table.


```python
jokes[0] = jokes[0].apply(lambda x: x.replace(':', ''))
jokes[0] = jokes[0].astype(int)
jokes.set_index(0, inplace=True)
```


```python
jokes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A man visits the doctor. The doctor says, "I h...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>This couple had an excellent relationship goin...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Q. What's 200 feet long and has 4 teeth? A. Th...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Q. What's the difference between a man and a t...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Q. What's O. J. Simpson's web address? A. Slas...</td>
    </tr>
  </tbody>
</table>
</div>



We will be creating a basic recommendation system using cosine similarity. 

Let's quickly review cosine similarity.

### Cosine similarity

Cosine similarty = 1 - cosign distance

#### What does cosine similarity measure?
- The angle between two vectors
    - if cosine(v1, v2) == 0 -> perpendicular
    - if cosine(v1, v2) == 1 -> same direction
    - if cosine(v1, v2) == -1 -> opposite direction

Let's create two vectors and find their cosine distance


```python
v1 = np.array([1, 2])
v2 = np.array([1, 2.5])

distance = cosine_distances(v1.reshape(1, -1), v2.reshape(1, -1))
```

Now, we can subtract the distance from 1 to find the cosine similarity.


```python
similarity = 1 - distance
similarity
```




    array([[0.99654576]])



There is also an function for this that we can use.


```python
cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))
```




    array([[0.99654576]])



# Build a recommender system 
How do we recommend a joke to userA?
- user to user ->
    - find users that are similar to userA
    - Identify jokes that have been rated highly by those similar users.

### Let's condition the data for a recommender system



```python
## User we would like to recommend a joke to
user_index = 0

## Drop column that totals the numbers of jokes each user has rated. 
## Isolate the row for the desired user
userA = df.drop(0, axis=1).loc[user_index, :]

# All other users
others = df.drop(0, axis=1).drop(index=user_index, axis=0)


# Find the nearest neighbors
knn = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
knn.fit(others)
```




    NearestNeighbors(algorithm='auto', leaf_size=30, metric='cosine',
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2, radius=1.0)



Great! Now we can use the vector of ratings for userA as an input to our knn model.

The knn model returns the distance between userA and the nearest K neighbors as well as their index.


```python
distances, indices = knn.kneighbors(userA.values.reshape(1, -1))
distances, indices = distances[0], indices[0]


print('---------------------------------------------------------------------------------------------')
print("userA's K nearest neighbor distances:", distances) 
print('---------------------------------------------------------------------------------------------')
print("Index for nearest neighbors indices:",indices)
print('---------------------------------------------------------------------------------------------')
```

    ---------------------------------------------------------------------------------------------
    userA's K nearest neighbor distances: [0.12284198 0.12953529 0.13661332 0.13848128 0.141326  ]
    ---------------------------------------------------------------------------------------------
    Index for nearest neighbors indices: [228 243 288 302  76]
    ---------------------------------------------------------------------------------------------


#### Now that we have our most similar users, what's next?

#### Find their highest rated items that aren't rated by userA


```python
# let's get jokes not rated by userA
jokes_not_rated = np.where(userA==99)[0]
jokes_not_rated = np.delete(jokes_not_rated, 0)
```

Next we need to isolate the nearest neighbors in our data, and examine their ratings for jokes userA has not rated.


```python
user_jokes = df.drop(0, axis=1).iloc[indices][jokes_not_rated]
user_jokes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>5</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>13</th>
      <th>27</th>
      <th>...</th>
      <th>140</th>
      <th>141</th>
      <th>142</th>
      <th>143</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>228</th>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-3.65625</td>
      <td>-10.00000</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-9.87500</td>
      <td>99.00000</td>
      <td>...</td>
      <td>99.0000</td>
      <td>99.000</td>
      <td>99.0000</td>
      <td>99.000</td>
      <td>99.0000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.000</td>
      <td>99.00000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>243</th>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-9.18750</td>
      <td>-6.43750</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>2.78125</td>
      <td>-3.09375</td>
      <td>...</td>
      <td>-5.1875</td>
      <td>-5.375</td>
      <td>-4.3125</td>
      <td>-4.125</td>
      <td>4.5625</td>
      <td>-2.9375</td>
      <td>-0.53125</td>
      <td>-3.875</td>
      <td>4.21875</td>
      <td>-4.875</td>
    </tr>
    <tr>
      <th>288</th>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-7.00000</td>
      <td>3.65625</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-6.81250</td>
      <td>6.37500</td>
      <td>...</td>
      <td>99.0000</td>
      <td>99.000</td>
      <td>99.0000</td>
      <td>99.000</td>
      <td>99.0000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.000</td>
      <td>99.00000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>302</th>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-9.03125</td>
      <td>9.56250</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>-9.43750</td>
      <td>-0.37500</td>
      <td>...</td>
      <td>99.0000</td>
      <td>99.000</td>
      <td>99.0000</td>
      <td>99.000</td>
      <td>99.0000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.000</td>
      <td>99.00000</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>76</th>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>3.03125</td>
      <td>-6.68750</td>
      <td>99</td>
      <td>99</td>
      <td>99</td>
      <td>4.09375</td>
      <td>-9.71875</td>
      <td>...</td>
      <td>99.0000</td>
      <td>99.000</td>
      <td>99.0000</td>
      <td>99.000</td>
      <td>99.0000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.000</td>
      <td>99.00000</td>
      <td>99.000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 87 columns</p>
</div>



Let's total up the ratings of each joke!

To do this, we need to replace 99 values with 0


```python
ratings = user_jokes.replace(99, 0).sum()
```

Right now, the user_jokes dataframe has rows set to individual users and jokes set as columns.

We want to look at the jokes of each of these users. To do that, let's transform our user_jokes dataframe


```python
user_jokes = user_jokes.T

user_jokes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>228</th>
      <th>243</th>
      <th>288</th>
      <th>302</th>
      <th>76</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>99.00000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>99.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>99.00000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>99.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>99.00000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>99.00000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-3.65625</td>
      <td>-9.1875</td>
      <td>-7.00000</td>
      <td>-9.03125</td>
      <td>3.03125</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-10.00000</td>
      <td>-6.4375</td>
      <td>3.65625</td>
      <td>9.56250</td>
      <td>-6.68750</td>
    </tr>
  </tbody>
</table>
</div>



Great! Now we add the joke ratings as a column to our user_jokes dataframe


```python
user_jokes['total'] = ratings
user_jokes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>228</th>
      <th>243</th>
      <th>288</th>
      <th>302</th>
      <th>76</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>99.00000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>99.00000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>99.00000</td>
      <td>99.0000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>99.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-3.65625</td>
      <td>-9.1875</td>
      <td>-7.00000</td>
      <td>-9.03125</td>
      <td>3.03125</td>
      <td>-25.84375</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-10.00000</td>
      <td>-6.4375</td>
      <td>3.65625</td>
      <td>9.56250</td>
      <td>-6.68750</td>
      <td>-9.90625</td>
    </tr>
  </tbody>
</table>
</div>



Using the method .idxmax(), we return the index for the joke with the highest rating!


```python
recommend_index = user_jokes['total'].idxmax()
recommend_index
```




    32




```python
# checking our work
user_jokes.sort_values(by='total', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>228</th>
      <th>243</th>
      <th>288</th>
      <th>302</th>
      <th>76</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>99.0</td>
      <td>2.31250</td>
      <td>7.21875</td>
      <td>-0.40625</td>
      <td>1.3750</td>
      <td>10.50000</td>
    </tr>
    <tr>
      <th>66</th>
      <td>99.0</td>
      <td>6.40625</td>
      <td>99.00000</td>
      <td>-0.56250</td>
      <td>3.4375</td>
      <td>9.28125</td>
    </tr>
    <tr>
      <th>54</th>
      <td>99.0</td>
      <td>-4.68750</td>
      <td>7.12500</td>
      <td>99.00000</td>
      <td>4.3125</td>
      <td>6.75000</td>
    </tr>
    <tr>
      <th>72</th>
      <td>99.0</td>
      <td>-1.50000</td>
      <td>6.00000</td>
      <td>-0.40625</td>
      <td>2.2500</td>
      <td>6.34375</td>
    </tr>
    <tr>
      <th>111</th>
      <td>99.0</td>
      <td>0.96875</td>
      <td>5.12500</td>
      <td>99.00000</td>
      <td>99.0000</td>
      <td>6.09375</td>
    </tr>
  </tbody>
</table>
</div>



Now all we have to do is plug in the index to our jokes dataframe, and return the recommended joke!


```python
jokes.iloc[recommend_index][1]
```




    'What do you call an American in the finals of the world cup? "Hey beer man!"'



# We did it!

### Assignment

Please create a function called `recommend_joke` that will receive a user index and returns a recommended joke.


```python
def recommendation_data():
    df = pd.read_csv('./data/jesterfinal151cols.csv', header=None)
    df = df.fillna(99)
    jokes = pd.read_table('./data/jester_items.tsv', header = None)
    jokes[0] = jokes[0].apply(lambda x: x.replace(':', ''))
    jokes[0] = jokes[0].astype(int)
    jokes.set_index(0, inplace=True)
    
    return df, jokes

def userA_and_others(user_index, df):
    ## Drop column that counts the numbers of jokes each user has rated. 
    ## Isolate the row for the desired user
    userA = df.drop(0, axis=1)\
          .loc[user_index, :]
    
    # Isolate all other users
    others = df.drop(0, axis=1).drop(index=user_index, axis=0)
    
    return userA, others

def nearest_neighbors(userA, others):
    # Fit Nearest Neighbors
    knn = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    knn.fit(others)
    
    distances, indices = knn.kneighbors(userA.values.reshape(1, -1))
    distances, indices = distances[0], indices[0] 
    
    return distances, indices

def find_joke(df, neighbor_indices, jokes_not_rated):
    
    user_jokes = df.drop(0, axis=1).iloc[neighbor_indices][jokes_not_rated]
    ratings = user_jokes.replace(99, 0).sum()
    user_jokes = user_jokes.T
    user_jokes['total'] = ratings
    recommend_index = user_jokes['total'].idxmax()
    return jokes.iloc[recommend_index][1]    

def recommend_joke(user_index):
    
    df, jokes = recommendation_data()

    userA, others = userA_and_others(user_index, df)

    distances, neighbor_indices = nearest_neighbors(userA, others)

    
    jokes_not_rated = np.where(userA==99)[0]
    jokes_not_rated = np.delete(jokes_not_rated, 0)
    
    return find_joke(df, neighbor_indices, jokes_not_rated)
```

Now we can recommend a joke to any user in the dataset!


```python
recommend_joke(400)
```




    "Q: How many programmers does it take to change a lightbulb? A: NONE! That's a hardware problem..."



Let's see what the highest rated joke is for User 400.


```python
highest_rated_joke_index = df.iloc[400].replace(99,0).drop(0).idxmax()
print(jokes.iloc[highest_rated_joke_index].values[0])
```

    A country guy goes into a city bar that has a dress code, and the maitre d' demands he wear a tie. Discouraged, the guy goes to his car to sulk when inspiration strikes: He's got jumper cables in the trunk! So he wraps them around his neck, sort of like a string tie (a bulky string tie to be sure) and returns to the bar. The maitre d' is reluctant, but says to the guy, "Okay, you're a pretty resourceful fellow, you can come in... but just don't start anything!"

