'''
<-----------------------problem------------------------>
A company has prepared 10 diff version of ads for their product but they don't
know which is best to put on social network so they hire a data scientist to know which version of ad
will get max no of clicks or which is the best ad for user

<-----------------------------goal--------------------------->
we as a data scientist have to find the version of ad that will get most clicks so 
that company will show only that ad to all the users bcos they cant show all ads
due to limited budget

<------------------------------dataset----------------------------------->
 
our data is the data of simulation(what is going to happen when we show the ads to user)
ie. which version of the ad the user is going to click on) this  is sth what god knows 
so basically we start with no data

<-------------------------------strategy------------------------------------->
now what happens in real life is that we are are going to start experimenting with
diff versions of ads by placing them in the social network.


and according to the result we observed we will change our strategy to place these
ads on social network.
Now for this steps:
    each time a user login to his account we show them one version of these 10 ads
    and we will observe his response if user click on the ad we get a reward of 1 otherwise 0.
    and we are going to do this for 10000 diff users.
    however we are not going to show the diff version of the ads to each user randomly
    there is a specific strategy(Upper bound confidence).
    the key thing to reinforcement learning is that
    strategy will dpd at each round on the previous results we observed in previous rounds
    for eg. if we are at the 10th round then algorithm will look at the results
    observed during first 10 rounds and according to these results it will
    decide which version of ad is to be shown to user
    


'''


#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset 

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing the UCB
import math
d=10
N=10000

number_of_selections = [0]*d  #number of times ad i was selected upto round n
sum_of_rewards = [0]*d  #sum of rewards of ad i upto round n
total_reward = 0
ads_selected=[]

'''
At each round n we consider two numbers for each ad i(number of times ad i was selected upto round n)
number_of_selections[i] and sum_of_rewards[i](sum of rewards of ad i upto round n )
for these two numbers:
we compute average_reward(average reward ad i upto round n) and delta_i

now finally we select the ad that has max ucb(= average_reward + delta_i)

since initially we have no data so we are saying that for ist 10 rounds:
    we select ad1 for ist round,ad2 for 2nd round and so on upto 10th round
    after that we are selecting based on our UCB algo logic
'''

for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if number_of_selections[i]> 0:
           average_reward = sum_of_rewards[i]/number_of_selections[i]
           delta_i = math.sqrt(1.5*math.log(n)/number_of_selections[i])
           upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400  #very large number
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad]+=1
    reward = dataset.values[n,ad] # take the value of reward from the dataset
    sum_of_rewards[ad]+=reward
    total_reward+=reward
    

plt.hist(ads_selected)
plt.title('histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('number of times the ad was selected')
plt.show()