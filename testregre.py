import random
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def input_features_labels(filename):
    features = []
    weights = []
    durations = []
    with open(filename) as f:
        for line in f:
            a = list(map(float, line.split()))
            features.append(a[:-2])
            weights.append(a[-2])
            durations.append(a[-1])
    return (features, weights, durations)

FEATURES_FILE = 'features.txt'
features, weights, durations = input_features_labels(FEATURES_FILE)


def shuffle_data(features, weights, durations):
    data = list(zip(features, weights, durations))
    random.shuffle(data)
    return list(zip(*data))

def split_train_and_test(features, weights, durations):
    train_size = int(len(features) * 0.66)
    return ((features[:train_size], weights[:train_size], durations[:train_size]),
            (features[train_size:], weights[train_size:], durations[train_size:]))

def calc_error(weight, duration, correct_weight, correct_duration):
    INVERSE = np.array([
        [3554.42,  -328.119],
        [-328.119,  133.511]
        ])
    diff = np.array([duration - correct_duration, weight - correct_weight])
    return np.dot(np.dot(diff.transpose(), INVERSE), diff)

def calc_sse0(test_data):
    weights = test_data[1]
    durations = test_data[2]
    ave_w = np.average(weights)
    ave_d = np.average(durations)
    e = 0
    for w, d in zip(weights, durations):
        e += calc_error(ave_w, ave_d, w, d)
    return e

def calc_sse(results):
    e = 0
    for w, d, cw, cd in results:
        e += calc_error(w, d, cw, cd)
    return e

def do_test(train_data, test_data):
    def build(a, b):
        from sklearn import grid_search
        params = {
                'min_samples_split': [2, 4, 8, 16, 32, 64]
                }
        return grid_search.GridSearchCV(RandomForestRegressor(n_estimators=300), params).fit(a, b)
#         return RandomForestRegressor(n_estimators=100, min_samples_split=30).fit(a, b)

    weight_rf = build(train_data[0], train_data[1])
    duration_rf = build(train_data[0], train_data[2])

    results = []
    for feature, weight, duration in zip(*test_data):
        w = float(weight_rf.predict(feature))
        d = float(duration_rf.predict(feature))
        results.append((w, d, weight, duration))

    sse0 = calc_sse0(test_data)
    sse = calc_sse(results)
    score = 1e6 * max(0, 1 - sse / sse0)
    return score

scores = []
for _ in range(100):
    features, weights, durations = shuffle_data(features, weights, durations)
    train_data, test_data = split_train_and_test(features, weights, durations)
    score = do_test(train_data, test_data)
    print(score)
    scores.append(score)
print('average: {}'.format(np.average(scores)))
