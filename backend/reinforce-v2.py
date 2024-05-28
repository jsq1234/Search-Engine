import numpy as np
import random
import tensorflow as tf
from collections import deque
import heapq

class DeepQNetwork(tf.keras.Model):
    def _init_(self, n_documents, hidden_size=64):
        super(DeepQNetwork, self)._init_()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(n_documents, activation='linear')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x

class DQNAgent:
    def _init_(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DeepQNetwork(action_size)
        self.target_model = DeepQNetwork(action_size)
        self.replay_memory = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.99
        self.update_target_freq = 10
        self.target_update_counter = 0
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.trainable_variables = {}
        # Initialize action counts for UCB exploration
        self.action_counts = np.zeros(action_size)
        self.total_actions = 0  # Total number of actions taken
        self.ucb_c = 2.0  # UCB exploration constant
        self.q_values = np.zeros(action_size)
    def initialize_trainable_variables(self, query):
        self.trainable_variables[query] = [tf.Variable(tf.random_normal_initializer()(v.shape)) for v in self.model.trainable_variables]
        
    def choose_actions(self, scores, k):
        ucb_values = np.zeros(self.action_size)
        for i in range(self.action_size):
            if self.action_counts[i] == 0:
                # If action has not been explored, choose it with maximum priority
                return np.argsort(scores)[-k:][::-1]

            # Calculate UCB value for each action
            ucb_values[i] = self.q_values[i] + self.ucb_c * np.sqrt(np.log(self.total_actions) / self.action_counts[i])

        # Choose actions with maximum UCB values
        actions = np.argsort(ucb_values)[-k:][::-1]
        return actions

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.replay_memory) < self.batch_size:
            return

        minibatch = random.sample(self.replay_memory, self.batch_size)

        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model(np.array([next_state], dtype=np.float32)))

            q_values = self.model(np.array([state], dtype=np.float32))
            q_values = q_values.numpy()
            q_values[0][action] = target

            states.append(state)
            targets.append(q_values[0])

        states = np.array(states, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            loss = self.loss_fn(targets, q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.target_update_counter > self.update_target_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        else:
            self.target_update_counter += 1

    def decay_epsilon(self):
        self.epsilon = max(0.1, self.epsilon - self.epsilon_decay)

def get_initial_relevance_scores(documents, query):

    return np.random.rand(len(documents))

def update_document_scores(scores, actions, user_feedback):
    num_documents = len(scores)


    for action, feedback in zip(actions, user_feedback):
        if feedback == 0:  # Negative feedback, push document down
            # Calculate the score to push the document behind half of the total number of documents
            new_score = np.sort(scores)[num_documents // 2] - 0.1
            if new_score >= scores[action]:
                new_score = scores[action] - 0.1
            scores[action] = new_score
    return scores


def encode_query(query, num_queries):
    encoded_query = np.zeros(num_queries)
    try:
        query_index = int(query)
        if query_index < num_queries:
            encoded_query[query_index] = 1
    except ValueError:
        pass  # If query cannot be converted to integer, leave encoded_query as zeros
    return encoded_query

def run(documents, k):
    n_documents = len(documents)
    action_size = n_documents
    state_size = n_documents + 1  # Size of state: 1 for search word + n_documents for relevance scores
    agent = DQNAgent(state_size, action_size)
    query_scores_map = {}  

    while True:
        query = input("Enter your search query: ")
        if query not in query_scores_map:

            query_scores_map[query] = get_initial_relevance_scores(documents, query)

        scores = query_scores_map[query]
        actions = agent.choose_actions(scores, k)

        top_k_indices = heapq.nlargest(k, range(n_documents), key=lambda i: scores[i])
        top_k_documents = [documents[i] for i in top_k_indices]
        print("Top-k relevant documents:", top_k_documents)

        user_feedback = np.zeros(n_documents, dtype=int)
        for i, doc_id in enumerate(top_k_documents):
            user_feedback[i] = int(input(f"Is document {doc_id} relevant? (1 for yes, 0 for no): "))
        

        agent.total_actions += 1
        for action in actions:
            agent.action_counts[action] += 1

        scores = update_document_scores(scores, actions, user_feedback)
        query_scores_map[query] = scores


        agent.remember(scores, actions, user_feedback, next_state=scores, done=False)
        agent.train_model()
        agent.decay_epsilon()


        for i, feedback in enumerate(user_feedback):
            if feedback == 0 and i < len(actions):  
                query_scores_map[query][actions[i]] -= 0.1
                agent.remember(scores, actions[i], -0.1, next_state=scores, done=False)

if __name__ == "__main__":

    documents = [i for i in range(100)]
    run(documents, k=5)