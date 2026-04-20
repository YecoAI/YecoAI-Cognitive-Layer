import math
import json
import os

class CognitiveModel:
    def __init__(self, weights=None):
        self.classes_ = ["Normal", "Loop", "Amnesia"]
        self.coefs_ = None
        self.intercepts_ = None
        self.is_fitted = False
        if weights:
            self.coefs_ = weights['coefs']
            self.intercepts_ = weights['intercepts']
            self.classes_ = weights['classes']
            self.is_fitted = True

    def _dot(self, a, b):
        result = [0.0] * len(b[0])
        for i in range(len(a)):
            for j in range(len(b[0])):
                result[j] += a[i] * b[i][j]
        return result

    def _add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def _relu(self, a):
        return [max(0, x) for x in a]

    def predict(self, feature_vector, features_dict=None):
        if features_dict is not None:
            repetition = features_dict.get("repetition_score", 0)
            struct_loop = features_dict.get("struct_loop_flag", 0)
            ngram_repeat = features_dict.get("max_ngram_repeat", 0)
            length = features_dict.get("length", 0)
            normal_density = features_dict.get("normal_pattern_density", 0)
            unique_ratio = features_dict.get("unique_ratio", 0)
            persistence = features_dict.get("keyword_persistence", 1.0)
            entropy = features_dict.get("entropy", 0)
            stop_ratio = features_dict.get("stop_word_ratio", 0)
            num_pk = features_dict.get("num_prompt_keywords", 0)
            punc_density = features_dict.get("punc_density", 0)
            
            # 1. Detection of Loop (highest priority)
            is_repetitive = repetition > 0.7 or struct_loop > 0.3 or ngram_repeat > 0.35 or punc_density > 0.4
            if is_repetitive:
                if (persistence > 0.5 or (normal_density > 0.25 and persistence > 0.2)) and length < 3.0:
                      return "Normal", {"Normal": 0.95, "Loop": 0.03, "Amnesia": 0.02}
                return "Loop", {"Normal": 0.01, "Loop": 0.98, "Amnesia": 0.01}

            # 2. Detection of Amnesia
            is_structurally_poor = unique_ratio < 0.3 or entropy < 1.0 or stop_ratio < 0.05
            
            if persistence < 0.1 and num_pk >= 3:
                if normal_density < 0.3 or is_structurally_poor:
                    return "Amnesia", {"Normal": 0.05, "Loop": 0.05, "Amnesia": 0.90}
            
            if persistence < 0.2 and num_pk >= 2 and normal_density < 0.15:
                 return "Amnesia", {"Normal": 0.10, "Loop": 0.10, "Amnesia": 0.80}
            
            if features_dict.get("word_salad_score", 0) > 0.8 and normal_density < 0.1:
                return "Amnesia", {"Normal": 0.05, "Loop": 0.05, "Amnesia": 0.90}

            # 3. Detection of Normal
            if normal_density > 0.1 or persistence > 0.1 or unique_ratio > 0.4 or length < 1.2:
                if repetition < 0.8:
                    return "Normal", {"Normal": 0.99, "Loop": 0.0, "Amnesia": 0.01}
            
            # Default fallback for heuristics confidence
            return "Normal", {"Normal": 0.90, "Loop": 0.05, "Amnesia": 0.05}

        if not self.is_fitted:
            return "Normal", {"Normal": 1.0, "Loop": 0.0, "Amnesia": 0.0}

        layer_input = feature_vector[:22]
        for i in range(len(self.coefs_)):
            z = self._add(self._dot(layer_input, self.coefs_[i]), self.intercepts_[i])
            if i < len(self.coefs_) - 1:
                layer_input = self._relu(z)
            else:
                scores = z
        
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [s / sum_exp for s in exp_scores]
        
        max_prob_idx = probs.index(max(probs))
        prediction = self.classes_[max_prob_idx]
        scores_dict = {cls: float(prob) for cls, prob in zip(self.classes_, probs)}
        
        return prediction, scores_dict

    @classmethod
    def load_from_json(cls, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                weights = json.load(f)
            return cls(weights)
        return cls()
