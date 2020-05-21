# Written by Nishant Mittal aka nishantwrp

# Constants
LABELS = 7

# Main Code Begins Here
def prepare_observation_matrix(predicted_probs):
    matrix = [[0 for _ in range(len(predicted_probs))] for _ in range(LABELS)]

    for i, component_probs in enumerate(predicted_probs):
        for j, prob in enumerate(component_probs):
            matrix[j][i] = prob

    return matrix


class HMM():
    def __init__(self, transition_matrix, observation_matrix):
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix

    def predict_state_sequence(self, components):
        # Implementaion of verterbi algorithm
        dp = [[0 for _ in range(LABELS)] for _ in range(components)]

        for i in range(components):
            if i == 0:
                for j in range(LABELS):
                    dp[i][j] = self.observation_matrix[j][i]
            else:
                for j in range(LABELS):
                    all_probs = list()

                    for k in range(LABELS):
                        all_probs.append((dp[i-1][k]*self.transition_matrix[k][j])*self.observation_matrix[j][i])

                    dp[i][j] = max(all_probs)

        final_seq = [0 for _ in range(components)]

        for i in range(components):
            final_seq[i] = dp[i].index(max(dp[i]))

        return final_seq