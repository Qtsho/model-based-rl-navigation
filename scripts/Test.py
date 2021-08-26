import numpy as np
from typing_extensions import TypedDict
from typing import Tuple, List

# divide = '-------------------------------------------'

# #dimensions (num_sequences, horizon, self.ac_dim) in the range
# a = np.array([1,3,4,1,5,6])
# np.concatenate
# #print (a[-5:])

# '''random shooting method MPC'''
# a= 5 + np.random.random_sample(
#     (3, 10, 5)) * (10 - 5)
# #print (a)

# '''Calculate mean accross axis'''
# a = np.array([[1,3,4,2],[3,4,5,3]])
# b = np.array([[2,1,2,22],[4,4,5,33]])
# x = np.mean(a,axis=0)
# #print (x,a.shape)


# '''sample from a random uniform distribution'''

# min  = 1
# max = 5
# sample = (max - min)* np.random.random_sample((2,4)) + min
# print (sample)

# print (divide)
# '''sample from a normal distributions'''
# #np.random.normal(mean, standar variance etc.) or np.random.randn(), later is standart normal distribution
# #N(3,6.25)
# mean = 3 
# sigma = 2.5
# sampleN = mean + sigma*np.random.randn(2,4)
# print (sampleN)


# print (divide)

# a = np.array([1,3,3,2]) 
# b  = 10 
# print (*a)

# print (*np.arange(0,5,1))
# print (divide)
# dict = {
#     "name" : 'tien',
#     "age" :24

# }
# dict2 = {
#     "name" : 'Nicole',
#     "age" :21

# }
# #item['age'] for item in list
# print (dict['name'])
# list = [dict,dict2]
# print(list)

# print ([item['age'] for item in list]) # list comprehension

# ##
# print ("Huy question!!")
# mylist = [[1,2,3] , [4,5,6]]
# extra = [10,15]
# newlist = mylist
# [newlist[i].append(z) for i ,z in zip(range(len(mylist)), extra)]
# print (newlist)


# class PathDict(TypedDict):
#     observation: np.ndarray
#     reward: np.ndarray
#     action: np.ndarray
#     next_observation: np.ndarray
#     terminal: np.ndarray

# def Path(
#     obs: List[np.ndarray],
#     acs: List[np.ndarray],
#     rewards: List[np.ndarray],
#     next_obs: List[np.ndarray], 
#     terminals: List[bool],
# ) -> PathDict:


#     """
#         Take info (separate arrays) from a single rollout
#         and return it in a single dictionary
#     """
#     return {"observation" : np.array(obs, dtype=np.float32),
#             "reward" : np.array(rewards, dtype=np.float32),
#             "action" : np.array(acs, dtype=np.float32),
#             "next_observation": np.array(next_obs, dtype=np.float32),
#             "terminal": np.array(terminals, dtype=np.float32)}


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--verbose", help="increase output verbosity",
#                     action="store_true")
# args = parser.parse_args()

# if args.verbose:
#     print("verbosity turned on")

# path= {"obs": np.array([1,2,4]), "action": np.array([12,1,2])}

# path2= {"obs": np.array([1,1,1]),
#         "action": np.array([12,11,2])
#         }
# list = [path]
# list.append(path2)

# print (path["obs"])
##need a funtion convert list of rollouts into a single np array, using list comprehension 
a = np.array([1,2,4,4])
b = a, True
print (b)
#a = np.append(a,[1])

