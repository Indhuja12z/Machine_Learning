import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('example.csv')

concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of Specific Hypothesis:")
    print(specific_h)

    general_h = [["?" for _ in range(len(specific_h))]
                 for _ in range(len(specific_h))]
    print("Initialization of General Hypothesis:")
    print(general_h)

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            print("\nPositive example")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if target[i] == "no":
            print("\nNegative example")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("Step", i + 1)
        print("Specific:", specific_h)
        print("General:", general_h)

    # Remove fully general hypotheses
    general_h = [g for g in general_h if g != ['?', '?', '?', '?', '?', '?']]

    return specific_h, general_h


def predict(example, specific_h, general_h):
    # Check against Specific hypothesis
    for i in range(len(specific_h)):
        if specific_h[i] != '?' and example[i] != specific_h[i]:
            return "NO"

    # Check against General hypotheses
    for g in general_h:
        match = True
        for i in range(len(g)):
            if g[i] != '?' and example[i] != g[i]:
                match = False
                break
        if match:
            return "YES"

    return "NO"


# Training
s_final, g_final = learn(concepts, target)

print("\nFinal Specific Hypothesis:")
print(s_final)

print("\nFinal General Hypothesis:")
print(g_final)

# -------- PREDICTION PART --------
test_example = ['sunny', 'warm', 'normal', 'strong', 'warm', 'same']

result = predict(test_example, s_final, g_final)

print("\nTest Example:", test_example)
print("Prediction:", result)

