package yecoai

import (
	"encoding/json"
	"math"
	"os"
)

type CognitiveModel struct {
	Classes     []string
	Coefs       [][][]float64
	Intercepts  [][]float64
	IsFitted    bool
}

type Weights struct {
	Coefs      [][][]float64 `json:"coefs"`
	Intercepts [][]float64   `json:"intercepts"`
	Classes    []string      `json:"classes"`
}

func NewCognitiveModel(weights *Weights) *CognitiveModel {
	if weights != nil {
		return &CognitiveModel{
			Classes:     weights.Classes,
			Coefs:       weights.Coefs,
			Intercepts:  weights.Intercepts,
			IsFitted:    true,
		}
	}
	return &CognitiveModel{
		Classes:  []string{"Normal", "Loop", "Amnesia"},
		IsFitted: false,
	}
}

func (cm *CognitiveModel) dot(a []float64, b [][]float64) []float64 {
	result := make([]float64, len(b[0]))
	for i := range a {
		for j := range b[0] {
			result[j] += a[i] * b[i][j]
		}
	}
	return result
}

func (cm *CognitiveModel) add(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func (cm *CognitiveModel) relu(a []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		if a[i] > 0 {
			result[i] = a[i]
		} else {
			result[i] = 0
		}
	}
	return result
}

type PredictionResult struct {
	Prediction string
	Scores     map[string]float64
}

func (cm *CognitiveModel) Predict(featureVector []float64, featuresDict map[string]float64) *PredictionResult {
	if featuresDict != nil {
		repetition := featuresDict["repetition_score"]
		structLoop := featuresDict["struct_loop_flag"]
		ngramRepeat := featuresDict["max_ngram_repeat"]
		length := featuresDict["length"]
		normalDensity := featuresDict["normal_pattern_density"]
		uniqueRatio := featuresDict["unique_ratio"]
		persistence := featuresDict["keyword_persistence"]
		entropy := featuresDict["entropy"]
		stopRatio := featuresDict["stop_word_ratio"]
		numPk := featuresDict["num_prompt_keywords"]
		puncDensity := featuresDict["punc_density"]

		isRepetitive := repetition > 0.7 || structLoop > 0.3 || ngramRepeat > 0.35 || puncDensity > 0.4
		if isRepetitive {
			if (persistence > 0.5 || (normalDensity > 0.25 && persistence > 0.2)) && length < 3.0 {
				return &PredictionResult{
					Prediction: "Normal",
					Scores: map[string]float64{"Normal": 0.95, "Loop": 0.03, "Amnesia": 0.02},
				}
			}
			return &PredictionResult{
				Prediction: "Loop",
				Scores: map[string]float64{"Normal": 0.01, "Loop": 0.98, "Amnesia": 0.01},
			}
		}

		isStructurallyPoor := uniqueRatio < 0.3 || entropy < 1.0 || stopRatio < 0.05

		if persistence < 0.1 && numPk >= 3 {
			if normalDensity < 0.3 || isStructurallyPoor {
				return &PredictionResult{
					Prediction: "Amnesia",
					Scores: map[string]float64{"Normal": 0.05, "Loop": 0.05, "Amnesia": 0.90},
				}
			}
		}

		if persistence < 0.2 && numPk >= 2 && normalDensity < 0.15 {
			return &PredictionResult{
				Prediction: "Amnesia",
				Scores: map[string]float64{"Normal": 0.10, "Loop": 0.10, "Amnesia": 0.80},
			}
		}

		if wordSaladScore, ok := featuresDict["word_salad_score"]; ok && wordSaladScore > 0.8 && normalDensity < 0.1 {
			return &PredictionResult{
				Prediction: "Amnesia",
				Scores: map[string]float64{"Normal": 0.05, "Loop": 0.05, "Amnesia": 0.90},
			}
		}

		if normalDensity > 0.1 || persistence > 0.1 || uniqueRatio > 0.4 || length < 1.2 {
			if repetition < 0.8 {
				return &PredictionResult{
					Prediction: "Normal",
					Scores: map[string]float64{"Normal": 0.99, "Loop": 0.0, "Amnesia": 0.01},
				}
			}
		}

		return &PredictionResult{
			Prediction: "Normal",
			Scores: map[string]float64{"Normal": 0.90, "Loop": 0.05, "Amnesia": 0.05},
		}
	}

	if !cm.IsFitted {
		return &PredictionResult{
			Prediction: "Normal",
			Scores: map[string]float64{"Normal": 1.0, "Loop": 0.0, "Amnesia": 0.0},
		}
	}

	layerInput := featureVector[:22]
	for i := range cm.Coefs {
		z := cm.add(cm.dot(layerInput, cm.Coefs[i]), cm.Intercepts[i])
		if i < len(cm.Coefs)-1 {
			layerInput = cm.relu(z)
		} else {
			scores := z

			maxScore := scores[0]
			for _, s := range scores {
				if s > maxScore {
					maxScore = s
				}
			}

			expScores := make([]float64, len(scores))
			sumExp := 0.0
			for i, s := range scores {
				expScores[i] = math.Exp(s - maxScore)
				sumExp += expScores[i]
			}

			probs := make([]float64, len(expScores))
			maxProbIdx := 0
			maxProb := 0.0
			for i, es := range expScores {
				probs[i] = es / sumExp
				if probs[i] > maxProb {
					maxProb = probs[i]
					maxProbIdx = i
				}
			}

			prediction := cm.Classes[maxProbIdx]
			scoresDict := make(map[string]float64)
			for i, cls := range cm.Classes {
				scoresDict[cls] = probs[i]
			}

			return &PredictionResult{
				Prediction: prediction,
				Scores:     scoresDict,
			}
		}
	}

	return &PredictionResult{
		Prediction: "Normal",
		Scores:     map[string]float64{"Normal": 1.0, "Loop": 0.0, "Amnesia": 0.0},
	}
}

func LoadModelFromJSON(path string) (*CognitiveModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return NewCognitiveModel(nil), err
	}

	var weights Weights
	err = json.Unmarshal(data, &weights)
	if err != nil {
		return NewCognitiveModel(nil), err
	}

	return NewCognitiveModel(&weights), nil
}

func GetDefaultModel() (*CognitiveModel, error) {
	return LoadModelFromJSON("weights.json")
}
