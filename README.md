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

### Joke data


```python
jokes = pd.read_table('./data/jester_items.tsv', header = None)
jokes.head()
```

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

There is also an function for this that we can use.


```python
cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))
```

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

Great! Now we add the joke ratings as a column to our user_jokes dataframe


```python
user_jokes['total'] = ratings
user_jokes.head()
```

Using the method .idxmax(), we return the index for the joke with the highest rating!


```python
recommend_index = user_jokes['total'].idxmax()
recommend_index
```


```python
# checking our work
user_jokes.sort_values(by='total', ascending=False).head()
```

Now all we have to do is plug in the index to our jokes dataframe, and return the recommended joke!


```python
jokes.iloc[recommend_index][1]
```

# We did it!

### Assignment

Please create a function called `recommend_joke` that will receive a user index and a knn model and returns a recommended joke.


```python
def recommend_joke(user_index, model):
    pass
```

Now we can recommend a joke to any user in the dataset!


```python
recommend_joke(400)
```

Let's see what the highest rated joke is for User 400.


```python
highest_rated_joke_index = df.iloc[400].replace(99,0).drop(0).idxmax()
print(jokes.iloc[highest_rated_joke_index].values[0])
```


```python

```
