import random
from collections import deque
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import os


class ModifiedTensorBoard(TensorBoard):
    # Override init
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class AI_Cruise:
    def __init__(self, LEARNING_RATE=0.001, DISCOUNT=0.99,
                 MINIBATCH_SIZE=64, REPLAY_MEMORY_SIZE=50000,
                 UPDATE_COUNTER=5, checkpoint_path='training_2/cp-{}.ckpt'):
        # Main model
        self.NAME = 'AI_CRUISE'

        self.learning_rate = LEARNING_RATE
        self.discount = DISCOUNT
        self.minibatch_size = MINIBATCH_SIZE
        self.update_counter = UPDATE_COUNTER
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.dirname(checkpoint_path)

        self.model = self.create_model()
        try:
            latest_weights = tf.train.latest_checkpoint(self.checkpoint_dir)
            self.model.load_weights(latest_weights)
        except:
            pass
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        now = datetime.now()
        now_format = now.strftime("%Y-%m-%dT%H-%M-%S")
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.NAME}-{now_format}")

        self.target_update_counter = 0
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                                      save_weights_only=True,
                                                                      verbose=1)
        # datetime.now()
        # now_format = now.strftime("%Y-%m-%dT%H-%M-%S")
        self.model.save_weights(checkpoint_path.format(date=datetime.now().strftime("%Y-%m-%dT%H-%M-%S")))

    def create_model(self):
        """
        Creates a model
        :return: The created model
        """

        print('*****************************************')
        print('*****************************************')
        # Create input layer
        m_input = tf.keras.layers.Input(shape=(11,), name='input')
        # Create hidden layer
        h_1 = tf.keras.layers.Dense(22, activation='relu', name='hidden_1')(m_input)
        h_2 = tf.keras.layers.Dense(18, activation='relu', name='hidden_2')(h_1)
        # Create multiple output layers
        out_1 = tf.keras.layers.Dense(4, activation='sigmoid', name='out_1')(h_2)
        out_2 = tf.keras.layers.Dense(1, activation='softmax', name='out_2')(h_2)
        out_3 = tf.keras.layers.Dense(2, activation='sigmoid', name='out_3')(h_2)
        # Create list of outputs
        outputs = [out_1, out_2, out_3]
        # Create model
        model = tf.keras.Model(inputs=m_input, outputs=outputs, name="AI_CRUISE")
        model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss='binary_crossentropy'
        )
        print(model.summary())

        print('*****************************************')
        print('*****************************************')

        return model

    def update_replay_memory(self, transition):
        """
        Adds the step data to a memory replay array
        :param transition: (observation_space, action, reward, new_observation_spoce, done)
        """
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.minibatch_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y1 = []
        y2 = []
        y3 = []

        # Enumerate batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # current_steering = current_qs_list[0][index]
            # current_gear = current_qs_list[1][index]
            # current_flaps = current_qs_list[2][index]

            future_steering = future_qs_list[0][index]
            future_gear = future_qs_list[1][index]
            future_flaps = future_qs_list[2][index]

            # Update
            if not done:
                update_steering = np.dot((reward + self.discount), action[0])
                update_gear = np.dot((reward + self.discount), action[1])
                update_flaps = np.dot((reward + self.discount), action[2])
            #     max_future_q = np.max(future_qs_list[index])
            #     new_q = reward + DISCOUNT * max_future_qð
            #     print('max_fut_q: {} \nnew_q: {}'.format(max_future_q, new_q))
            #     return
            else:
                update_steering = np.dot(reward, action[0])
                update_gear = np.dot(reward, action[1])
                update_flaps = np.dot(reward, action[2])
            #     new_q = reward

            # current_qs = current_qs_list[index]
            # current_qs = (current_qs_list[0][index], current_qs_list[1][index], current_qs_list[2][index])
            # print('current_qs: {} \nnew_q: {} action: {}'.format(current_qs, new_q, action))
            # current_qs[action] = new_q
            # print('current_qs_list: {}'.format(current_qs_list[0]))
            X.append(current_state)
            # y1.append(current_qs_list[0][index])
            # y2.append(current_qs_list[1][index])
            # y3.append(current_qs_list[2][index])
            # y1.append(future_steering)
            # y2.append(future_gear)
            # y3.append(future_flaps)
            y1.append(update_steering)
            y2.append(update_gear)
            y3.append(update_flaps)

        self.model.fit(
            {"input": np.array(X)/255},
            {
                "out_1": np.array(y1),
                "out_2": np.array(y2),
                "out_3": np.array(y3)
            },
            batch_size=self.minibatch_size,
            verbose=0,
            shuffle=False)  #,
            # callbacks=[self.checkpoint_callback] if terminal_state else None)
            # callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter
        if terminal_state:
            self.target_update_counter += 1
            self.model.save_weights(self.checkpoint_path.format(date=datetime.now().strftime("%Y-%m-%dT%H-%M-%S")))

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.update_counter:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state, target_model=False):
        """
        Get Q values given current observation space (environment state)
        :param target_model: To be executed on the target model, default = False
        :param state: The current observation state
        :return: The predicted actions
        """

        state_array = np.array(state)
        # print('state_array: {}'.format(state_array))
        steering, gear, flaps = self.model.predict(state_array.reshape(-1, *state_array.shape)) \
            if target_model \
            else self.target_model.predict(state_array.reshape(-1, *state_array.shape))

        # Convert landing gear from float to int
        gear = gear.astype('int')
        # print('steering: {}'.format(steering))
        # print('gear: {}'.format(gear))
        # print('flaps: {}'.format(flaps))

        m_list = []
        for l in steering[0]:
            m_list.append(l)

        for l in gear[0]:
            m_list.append(l)

        for l in flaps[0]:
            m_list.append(l)

        return m_list
