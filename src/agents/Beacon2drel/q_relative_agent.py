from src.lib.tensorflow_agent import TensorflowAgent


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
import shutil

from absl import flags
flags.DEFINE_float("gamma", 0.9, "Współczynnik określający, jak ważne są przyszłe doświadczenia")
flags.DEFINE_float("lr", 1e-2, "Współczynnik określający szybkość uczenia")
flags.DEFINE_float("hidden", 8, "Współczynnik określający szybkość uczenia")



class QRelativeAgent(TensorflowAgent):
    """Test agent executing random ctions"""
    def __init__(self):
        TensorflowAgent.__init__(self)

        self.gamma = flags.FLAGS.gamma
        self.lr = flags.FLAGS.lr

        # parametry dla setup
        s_size = 64*64 # warstwa obserwacji
        h_size = 64*64 # warstwa ukryta
        a_size = 8 # warstwa akcji

        #feed forwards część
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32) # placeholder na bufor z obserwajcą
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu) # warstwa ukryta    
        #funkcja aktywacji: x > 0           Computes rectified linear: max(features, 0).
        self.output = slim.fully_connected(hidden,a_size,activation_fn =tf.nn.softmax,biases_initializer=None) # bufor z akcjami
        # softmax :/
        self.chosen_action = tf.argmax(self.output,1) # wybrana akcja, po największej wartości na wyjściu

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.


        # stary sposób uczenia
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)  # bufor na nagrodę
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)    # bufor na akcję
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        # utwórz tablicę o rozmiarze outputa, która zawiera indeksy wybranych akcji
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        # utwórz macierz o rozmiarze outputa, która zawiera 
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        #???
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)
        #oblicza pochodne cząstkowe? po funkcji loss,
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

    def setup(self):
        
        self.memory_buffer = [] # bufor na obserwacje
        #inicializuj tensorflow
        self.sess.run(tf.global_variables_initializer())    
        self.gradBuffer = self.sess.run(tf.trainable_variables())
        for ix,grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0
            

    def act(self, observation, action_space):
        sess = self.sess
        self.observation = observation
        #Probabilistically pick an action given our network outputs.
        a_dist = sess.run(self.output,feed_dict={self.state_in:[observation]})
        self.a_dist = a_dist
        
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a

    def observe(self, observation, reward, action):
        self.memory_buffer.append([self.observation, action, reward, observation])
        self.observation = observation
        
    def next_episode(self, episode):
        #Update the network.
        sess = self.sess

        if len(self.memory_buffer) == 0:
            return

        memory_buffer = np.array(self.memory_buffer)
        np.random.shuffle(memory_buffer)
        self.memory_buffer = []
        memory_buffer[:,2] = self.discount_rewards(memory_buffer[:,2])
         # posumuj nagrody mniejszając znaczenia nagrody wraz z kolejnymi akcjami
        feed_dict={# przekształcenie bufora w słownik
                self.reward_holder:memory_buffer[:,2],
                self.action_holder:memory_buffer[:,1],
                self.state_in:np.vstack(memory_buffer[:,0])}

        grads = sess.run(self.gradients, feed_dict=feed_dict)
        for idx,grad in enumerate(grads):
            self.gradBuffer[idx] += grad

        if episode % 5 == 0 and episode != 0:
            feed_dict= dictionary = dict(zip(self.gradient_holders, self.gradBuffer))
            _ = sess.run(self.update_batch, feed_dict=feed_dict)
            for ix,grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad * 0
    
    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        # sumujemy nagrody, pamietając stare ze współczynnikiem gamma
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def save(self, fileName):
        import os
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        else:
            shutil.rmtree(fileName)
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(self.sess, fileName)

    def load(self, fileName):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self.sess, fileName)
