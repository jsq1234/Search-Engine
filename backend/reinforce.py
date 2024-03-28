import numpy as np
import heapq


class DocumentRetrievalEnv:
    def __init__(self, ranked_documents, relevant_docs):
        self.n_documents = len(ranked_documents)
        self.documents = ranked_documents
        self.k = len(relevant_docs)
        # Assume document IDs are integers
        self.scores = ranked_documents['score']
        self.step_count = np.random.randint(0, 100)

    def reset(self):
        self.step_count = 0

    def step(self, actions, relevance_list):
        rewards = []
        for action in actions:
            document_id = self.documents['doc_id'][action]

            user_feedback = next(
                (rel for (doc_id, rel) in relevance_list if doc_id == document_id),
                None)
            # Simulate user feedback
            # user_feedback = int(
            #     input(f"Is document {document_id} relevant? (1 for yes, 0 for no): "))

            # Update score based on user feedback
            reward = 0
            if user_feedback == 0:
                reward = -10
            elif user_feedback == 1:
                reward = 5
            else:
                reward = 10

            self.documents['score'][action] += reward
            # scores[document_id] += reward
            rewards.append(self.documents['score'][action])

        # self.step_count += len(actions)
        # done = self.step_count >= self.n_documents  # Terminate after all documents are retrieved

        return rewards


class QLearningAgent:
    def __init__(self, n_documents):
        self.n_documents = n_documents
        self.q_table = np.zeros(n_documents)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1
        self.decay_rate = 0.15

    def choose_actions(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_documents, size=state, replace=False)
        else:
            return np.argsort(self.q_table)[-state:][::-1]

    def update_q_values(self, actions, rewards):
        for action, reward in zip(actions, rewards):
            self.q_table[action] += self.alpha * \
                (reward + self.gamma *
                 np.max(self.q_table) - self.q_table[action])
        # x = 0
        # for i in range(100):
        #     print(self.q_table[i])

    def decay_epsilon(self):
        self.epsilon = max(0.1, self.epsilon - self.decay_rate)


def run(documents, relevant_documents, relevance_list, k):
    n_documents = len(documents)
    env = DocumentRetrievalEnv(documents, relevant_documents)
    agent = QLearningAgent(n_documents)

    env.reset()
    x = [-1 for _ in range(k)]
    while True:
        # print(f"Iteration {env.step_count}")
        actions = agent.choose_actions(env.k)
        x.sort()
        actions.sort()
        # print(actions)
        flag = 0
        for i in range(min(k, env.k)):
            if (actions[i] != x[i]):
                flag = 1
                break
        if (flag == 0):
            break
        x = actions
        rewards = env.step(actions, relevance_list)
        agent.update_q_values(actions, rewards)
        agent.decay_epsilon()
        env.step_count += 1
        # state = env.get_next_state()

    # Now you can use the learned Q-values to retrieve top-k documents for a given state
    top_k_indices = heapq.nlargest(
        k, range(n_documents), key=lambda i: documents['score'][i])
    top_k_documents = documents[top_k_indices]

    return top_k_documents
    # top_k_documents = np.argsort(env.scores)[-k:][::-1]
    print("Top-k relevant documents:", top_k_documents)


if __name__ == "__main__":
    run()
