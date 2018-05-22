#Name: Xiangnan He
#ID: xhe321

"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def author(self):
        return 'xhe321'

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.s = 0
        self.a = 0
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions        
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q_table = (2*np.random.rand(self.num_states, self.num_actions)) - 1
        self.T1 = 0.00001*np.ones([self.num_states, self.num_actions, self.num_states])
        self.T2 = self.T1/(0.00001*self.num_states)
        self.R = np.zeros([self.num_states, self.num_actions])
        self.experience = []

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        
        rand_num = rand.random()
        action = None
        if rand_num < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            max_val = -float("inf")
            max_idx = None
            for i, val in enumerate(self.q_table[s]):
                if val > max_val:
                    max_val = val
                    max_idx = i
            action = max_idx
        if self.verbose: print "s =", s,"a =",action

        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        max_val = -float("inf")
        max_idx = None
        for i, val in enumerate(self.q_table[s_prime]):
            if max_val < val:
                max_val = val
                max_idx = i
        best_action = max_idx
        self.q_table[self.s][self.a] = (1-self.alpha)*self.q_table[self.s][self.a] + \
            self.alpha*(r + self.gamma * self.q_table[s_prime][best_action])

        if self.dyna > 0:
            self.T1[self.s, self.a, s_prime] = self.T1[self.s, self.a, s_prime] + 1.0
            self.T2[self.s, self.a, :] = self.T1[self.s, self.a,:]/self.T1[self.s, self.a, :].sum()
            self.R[self.s, self.a] = (1-self.alpha)*self.R[self.s, self.a] + self.alpha * r
            self.experience.append((self.s, self.a))

            for i in range(0, self.dyna):
                exp = rand.choice(self.experience)
                opt_action_idx = self.T2[exp[0], exp[1], :].argmax()
                r = self.R[exp[0], exp[1]]
                self.q_table[exp[0], exp[1]] = (1-self.alpha)*self.q_table[exp[0], exp[1]] +\
                    self.alpha* (r + self.gamma * self.q_table[opt_action_idx, :].max())


        self.s = s_prime

        rand_num = rand.random()
        if rand_num < self.rar:
            self.a = rand.randint(0, self.num_actions-1)
        else:
            max_val = -float("inf")
            max_idx = None
            for i, val in enumerate(self.q_table[self.s]):
                if max_val < val:
                    max_val = val
                    max_idx = i
            self.a = max_idx

        
        self.rar = self.rar * self.radr

        if self.verbose: print "s =", s_prime,"a =",self.a,"r =",r

        return self.a

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
