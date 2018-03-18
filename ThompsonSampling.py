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
    there is a specific strategy(Thompson Sampling).
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

#implementing the Thompson sampling
import random
d=10
N=10000

number_of_rewards_1 = [0]*d  
number_of_rewards_0 = [0]*d  
total_reward = 0
ads_selected=[]



for n in range(0,N):
    max_random = 0
    ad = 0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad] # take the value of reward from the dataset
    if reward==1:
        number_of_rewards_1[ad]+=1
    else:
         number_of_rewards_0[ad]+=1
        
    total_reward+=reward
    

plt.hist(ads_selected)
plt.title('histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('number of times the ad was selected')
plt.show()