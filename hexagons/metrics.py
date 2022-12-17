from sklearn.metrics import f1_score

from hexagons.board import COLOR2INDEX


class Metrics:
    def __init__(self):
        self.all_true_actions = []
        self.all_pred_actions = []
        self.all_true_states = []
        self.all_pred_states = []

    def add(self, true_actions, pred_actions, true_state, pred_state):
        true_actions.sort()
        pred_actions.sort()
        self.all_true_actions.append(true_actions[:])
        self.all_pred_actions.append(pred_actions[:])
        self.all_true_states.append(true_state[:])
        self.all_pred_states.append(pred_state[:])

    def actions_em(self):
        assert len(self.all_true_actions) == len(self.all_pred_actions)
        correct_cnt = sum(int(t == p) for t, p in zip(self.all_true_actions, self.all_pred_actions))
        return float(correct_cnt) / len(self.all_true_actions)

    def board_em(self):
        assert len(self.all_true_states) == len(self.all_pred_states)
        correct_cnt = sum(int(t == p) for t, p in zip(self.all_true_states, self.all_pred_states))
        return float(correct_cnt) / len(self.all_true_states)

    def calc_f1_sets(self, true_set, pred_set):
        if not true_set:
            return float(true_set == pred_set)
        if not pred_set:
            return 0.0
        intersection = set()
        for a in pred_set:
            if a in true_set:
                intersection.add(a)
        precision = float(len(intersection)) / len(pred_set)
        recall = float(len(intersection)) / len(true_set)
        if precision + recall <= 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def actions_f1(self):
        scores = []
        for true_actions, pred_actions in zip(self.all_true_actions, self.all_pred_actions):
            true_actions = set(true_actions)
            pred_actions = set(pred_actions)
            scores.append(self.calc_f1_sets(true_actions, pred_actions))
        return sum(scores) / len(scores)

    def board_f1(self):
        scores = []
        for true_state, pred_state in zip(self.all_true_states, self.all_pred_states):
            true_state = {(i, c) for i, c in enumerate(true_state) if c != 0}
            pred_state = {(i, c) for i, c in enumerate(pred_state) if c != 0}
            scores.append(self.calc_f1_sets(true_state, pred_state))
        return sum(scores) / len(scores)

    def print_all(self):
        print("Actions EM:", self.actions_em())
        print("Actions F1:", self.actions_f1())
        print("Board EM:", self.board_em())
        print("Board F1:", self.board_f1())
