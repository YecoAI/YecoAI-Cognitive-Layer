package main

import (
	"fmt"
	"yecoai-cognitive-layer/yecoai"
)

func main() {
	engine := yecoai.NewFeatureEngine()
	model, err := yecoai.GetDefaultModel()
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}

	fmt.Println("YecoAI Cognitive Layer - Go Version")
	fmt.Println("==================================")

	testCases := []struct {
		name   string
		text   string
		prompt string
	}{
		{
			name:   "Normal response",
			text:   "The quick brown fox jumps over the lazy dog. Photosynthesis is the process by which plants convert sunlight into energy.",
			prompt: "Write about photosynthesis and animals.",
		},
		{
			name:   "Loop example",
			text:   "and then and then and then and then and then and then and then and then and then and then and then and then and then and then",
			prompt: "Tell a story about a forest.",
		},
		{
			name:   "Amnesia example",
			text:   "asdf qwerty zxcv poiuy lkjh mnbv qwerty asdf zxcv",
			prompt: "Explain photosynthesis in detail, including how plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
		},
		{
			name:   "Very long loop",
			text:   "test test test test test test test test test test test test test test test test test test test test test test",
			prompt: "Write an essay about history.",
		},
	}

	for _, tc := range testCases {
		fmt.Printf("\n=== %s ===\n", tc.name)
		fmt.Printf("Text: %s\n", tc.text)

		result := engine.ExtractFeatures(tc.text, tc.prompt)
		prediction := model.Predict(result.Vector, result.Features)

		fmt.Printf("\nPrediction: %s\n", prediction.Prediction)
		fmt.Printf("Scores:\n")
		for class, score := range prediction.Scores {
			fmt.Printf("  %s: %.4f\n", class, score)
		}
		fmt.Printf("\nFeatures:\n")
		fmt.Printf("  Repetition score: %.4f\n", result.Features["repetition_score"])
		fmt.Printf("  Struct loop flag: %.4f\n", result.Features["struct_loop_flag"])
		fmt.Printf("  Keyword persistence: %.4f\n", result.Features["keyword_persistence"])
		fmt.Printf("  Semantic coherence: %.4f\n", result.Features["semantic_coherence"])
	}
}
