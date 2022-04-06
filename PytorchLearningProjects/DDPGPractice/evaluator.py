
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

import util 

class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path="", max_episode_length=None):
        self.num_episodes = num_episodes
        if max_episode_length is None:
            max_episode_length = 200
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)

    def __call__(self, env, policy, debug=False, save=True):
        
        self.is_training = False
        # observation = None
        all_observations = None
        result = []
        
        for episode in range(self.num_episodes):
            
            # reset at the start of episode
            all_observations = env.reset()
            episode_steps = 0
            episode_reward = 0
            # assert observation is not None
            assert all_observations is not None
            
            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                actions = {}
                for ship_id, ship_observation in all_observations.items():
                    actions[ship_id] = policy(ship_observation)
                
                all_observations, train_reward, done = env.step(**actions)
                if episode_steps >= self.max_episode_length -1:   #  一个回合结束的标志就是    步数达到最大值
                    done = True
                
                # update
                episode_reward += train_reward       # episode_reward 代表每一个回合的奖励
                episode_steps += 1
            # episode  回合    epiusode_reward  奖励
            if debug: util.prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)        # result   存储在 num_episodes    回合里面的所有奖励结
            
        result = np.array(result).reshape(-1,1)   # 不管分多少行，我需要分成一列            在这里相当于转置了一下，一行变一列
        self.results = np.hstack([self.results, result])  # 按照列合并             这个只是保存训练结果
        
        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)   # 返回多次训练结果的奖励平均值，越高越好

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
        
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn + '.png')
        savemat(fn+'.mat', {'reward':self.results})
        












