package yecoai

import (
	"math"
	"regexp"
	"strings"
)

type FeatureEngine struct {
	stopWords       map[string]bool
	resetKeywords   map[string]bool
	loopKeywords    map[string]bool
	normalPatterns  map[string]bool
}

func NewFeatureEngine() *FeatureEngine {
	stopWords := make(map[string]bool)
	stopWordsList := []string{
		"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "it", "this", "that", "you", "i", "me", "my",
		"can", "could", "will", "would", "shall", "should", "do", "does", "did", "have", "has", "had", "be", "been", "being", "am", "are", "was", "were",
		"what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
		"write", "explain", "describe", "summarize", "translate", "list", "show", "help", "please", "repetitive", "poem", "short", "story",
		"compose", "verse", "themed", "medical", "humor", "joke", "tell", "query", "response", "task", "instruction", "snippet", "basic", "casual", "generate", "nature",
		"numeric", "echo", "humour", "encouragement", "cheer", "repeated", "greeting", "poetic", "repetition",
		"et", "est", "non", "ad", "ut", "cum", "quod", "qui", "que", "sed", "si", "per", "ex", "de", "esse", "sunt",
		"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "也", "他", "一个", "说", "去", "谢谢", "我们", "你们", "他们",
		"di", "e", "il", "la", "che", "un", "una", "con", "su", "mi", "ti", "è", "ma", "ed", "se", "perché", "come", "molto", "sono", "ho", "ha", "abbiamo", "hanno",
	}
	for _, word := range stopWordsList {
		stopWords[word] = true
	}

	resetKeywords := make(map[string]bool)
	resetKeywordsList := []string{
		"reset", "forget", "clear", "wipe", "purge", "override", "nullify", "reboot",
		"dimentica", "resetta", "cancella", "vuoto", "ignora", "pulisci",
		"重置", "忘记", "清除", "清空", "重启",
	}
	for _, word := range resetKeywordsList {
		resetKeywords[word] = true
	}

	loopKeywords := make(map[string]bool)
	loopKeywordsList := []string{
		"loop", "ripeti", "repeat", "redundancy", "stuck", "bloccato", "endless",
		"循环", "重复", "卡住", "再次", "ancora", "infinito",
	}
	for _, word := range loopKeywordsList {
		loopKeywords[word] = true
	}

	normalPatterns := make(map[string]bool)
	normalPatternsList := []string{
		"hello", "hi", "thanks", "thank", "please", "could", "would", "how", "what", "can", "need", "write", "explain", "describe", "tell", "show", "give", "make",
		"happy", "sad", "good", "bad", "better", "best", "great", "awesome", "cool", "nice",
		"mist", "roll", "silent", "peak", "forest", "breath", "morning", "light",
		"patient", "doctor", "glass", "opinion", "reply", "funny",
		"photosynthesis", "sunlight", "carbon", "dioxide", "oxygen", "energy", "sugar", "chlorophyll", "conversion",
		"def", "add", "return", "function", "parameter", "snippet", "code", "programming",
		"ready", "assist", "request", "chat",
		"1984", "orwell", "george", "totalitarianism", "dystopian", "fiction", "novel", "surveillance", "aspect", "control",
		"capital", "france", "paris", "london", "rome", "berlin", "tokyo", "beijing", "washington", "city", "major", "hub",
		"Peak", "reach", "sky", "snow", "eagle", "fly",
		"salve", "ave", "gratias", "quomodo", "quis", "quid", "ubi", "quando", "ciao", "grazie", "per favore", "potresti", "spiega", "descrivi",
		"opera", "poesia", "storia", "scienza", "libro", "film", "musica", "viaggio", "cibo", "lavoro", "pomeriggio", "sera", "notte", "giorno",
		"你好", "谢谢", "请", "怎么", "什么", "哪里", "帮助", "解释", "描述", "分析", "研究", "讨论", "建议",
	}
	for _, word := range normalPatternsList {
		normalPatterns[word] = true
	}

	return &FeatureEngine{
		stopWords:       stopWords,
		resetKeywords:   resetKeywords,
		loopKeywords:    loopKeywords,
		normalPatterns:  normalPatterns,
	}
}

var (
	tokenizeRegex      = regexp.MustCompile(`\b\w+\b|\p{Han}|[^\w\s]`)
	digitRegex         = regexp.MustCompile(`\d`)
	puncRegex          = regexp.MustCompile(`[^\w\s\p{Han}]`)
	vowelRegex         = regexp.MustCompile(`[aeiouàèìòù]`)
	alphaRegex         = regexp.MustCompile(`[a-zàèìòù]`)
	binaryOrDigitRegex = regexp.MustCompile(`^[01\s]{10,}$|^\d{15,}$`)
)

func (fe *FeatureEngine) tokenize(text string) []string {
	tokens := tokenizeRegex.FindAllString(strings.ToLower(text), -1)
	result := make([]string, len(tokens))
	for i, t := range tokens {
		if len(t) > 3 && strings.HasSuffix(t, "s") {
			result[i] = t[:len(t)-1]
		} else {
			result[i] = t
		}
	}
	return result
}

func (fe *FeatureEngine) calculateEntropy(items []string) float64 {
	if len(items) == 0 {
		return 0.0
	}
	counts := make(map[string]int)
	for _, item := range items {
		counts[item]++
	}
	total := len(items)
	entropy := 0.0
	for _, count := range counts {
		prob := float64(count) / float64(total)
		entropy -= prob * math.Log2(prob)
	}
	return entropy
}

func (fe *FeatureEngine) calculateBurstiness(tokens []string) float64 {
	if len(tokens) < 10 {
		return 0.0
	}

	positions := make(map[string][]int)
	for i, token := range tokens {
		positions[token] = append(positions[token], i)
	}

	var intervals []float64
	for _, posList := range positions {
		if len(posList) > 1 {
			for i := 0; i < len(posList)-1; i++ {
				intervals = append(intervals, float64(posList[i+1]-posList[i]))
			}
		}
	}

	if len(intervals) == 0 {
		return 0.0
	}

	avgInt := 0.0
	for _, interval := range intervals {
		avgInt += interval
	}
	avgInt /= float64(len(intervals))

	variance := 0.0
	for _, interval := range intervals {
		variance += (interval - avgInt) * (interval - avgInt)
	}
	variance /= float64(len(intervals))

	stdDev := math.Sqrt(variance)

	cv := 0.0
	if avgInt > 0 {
		cv = stdDev / avgInt
	}

	result := (cv - 1) / (cv + 1)
	if result < 0 {
		result = 0
	}
	if result > 1 {
		result = 1
	}
	return result
}

func (fe *FeatureEngine) getKeywords(tokens []string) []string {
	var keywords []string
	digitOnlyRegex := regexp.MustCompile(`^\d+$`)
	for _, t := range tokens {
		if (len(t) >= 3 && !fe.stopWords[t]) || digitOnlyRegex.MatchString(t) {
			keywords = append(keywords, t)
		}
	}
	return keywords
}

func (fe *FeatureEngine) getCommonPrefix(s1, s2 string) bool {
	if strings.Contains(s1, s2) || strings.Contains(s2, s1) {
		return true
	}
	minLen := len(s1)
	if len(s2) < minLen {
		minLen = len(s2)
	}
	if minLen < 3 {
		return s1 == s2
	}
	common := 0
	for i := 0; i < minLen; i++ {
		if s1[i] == s2[i] {
			common++
		} else {
			break
		}
	}
	maxLen := len(s1)
	if len(s2) > maxLen {
		maxLen = len(s2)
	}
	return float64(common)/float64(maxLen) > 0.6
}

type FeatureResult struct {
	Vector   []float64
	Features map[string]float64
}

func (fe *FeatureEngine) ExtractFeatures(text, prompt string) *FeatureResult {
	if text == "" {
		vector := make([]float64, 25)
		features := map[string]float64{
			"length": 0, "unique_ratio": 0.0, "entropy": 0.0, "repetition_score": 0.0, "max_ngram_repeat": 0.0,
			"stop_word_ratio": 0.0, "alpha_ratio": 0.0, "avg_token_len": 0.0, "reset_flag": 0.0, "loop_keyword_flag": 0.0,
			"punc_density": 0.0, "vowel_ratio": 0.0, "normal_pattern_density": 0.0, "char_repetition": 0.0, "word_salad_score": 0.0, "token_diversity": 0.0,
			"entropy_2gram": 0.0, "adversarial_score": 0.0, "struct_loop_flag": 0.0, "semantic_coherence": 0.0,
			"salad_diff": 0.0, "digit_density": 0.0, "burstiness": 0.0, "keyword_persistence": 1.0,
		}
		return &FeatureResult{Vector: vector, Features: features}
	}

	var tokens []string
	if binaryOrDigitRegex.MatchString(strings.TrimSpace(text)) {
		tokens = []string{strings.TrimSpace(text)}
	} else {
		tokens = fe.tokenize(text)
	}

	numTokens := len(tokens)

	keywordPersistence := 1.0
	numPromptKeywords := 0
	if prompt != "" {
		promptTokens := fe.tokenize(prompt)
		promptKeywords := fe.getKeywords(promptTokens)
		numPromptKeywords = len(promptKeywords)
		if len(promptKeywords) > 0 {
			textTokensSet := make(map[string]bool)
			for _, t := range tokens {
				textTokensSet[t] = true
			}
			matches := 0
			for _, pk := range promptKeywords {
				if textTokensSet[pk] {
					matches++
				} else {
					found := false
					for _, tk := range tokens {
						if len(tk) >= 4 && fe.getCommonPrefix(pk, tk) {
							found = true
							break
						}
					}
					if found {
						matches++
					}
				}
			}
			keywordPersistence = float64(matches) / float64(len(promptKeywords))
		}
	}

	if numTokens == 0 {
		vector := make([]float64, 25)
		features := map[string]float64{
			"length": 0, "unique_ratio": 0.0, "entropy": 0.0, "repetition_score": 0.0, "max_ngram_repeat": 0.0,
			"stop_word_ratio": 0.0, "alpha_ratio": 0.0, "avg_token_len": 0.0, "reset_flag": 0.0, "loop_keyword_flag": 0.0,
			"punc_density": 0.0, "vowel_ratio": 0.0, "normal_pattern_density": 0.0, "char_repetition": 0.0, "word_salad_score": 0.0, "token_diversity": 0.0,
			"entropy_2gram": 0.0, "adversarial_score": 0.0, "struct_loop_flag": 0.0, "semantic_coherence": 0.0,
			"salad_diff": 0.0, "digit_density": 0.0, "burstiness": 0.0, "keyword_persistence": keywordPersistence,
			"num_prompt_keywords": float64(numPromptKeywords),
		}
		return &FeatureResult{Vector: vector, Features: features}
	}

	length := math.Log1p(float64(numTokens))

	uniqueTokens := make(map[string]bool)
	for _, t := range tokens {
		uniqueTokens[t] = true
	}
	uniqueRatio := float64(len(uniqueTokens)) / float64(numTokens)

	entropy := fe.calculateEntropy(tokens)
	burstiness := fe.calculateBurstiness(tokens)

	repetitionScore := 0.0
	if numTokens >= 3 {
		ngrams := make(map[string]int)
		for i := 0; i < numTokens-2; i++ {
			key := tokens[i] + "|" + tokens[i+1] + "|" + tokens[i+2]
			ngrams[key]++
		}
		repeatedNgrams := 0
		for _, count := range ngrams {
			if count > 1 {
				repeatedNgrams += count
			}
		}
		if (numTokens - 2) > 0 {
			repetitionScore = float64(repeatedNgrams) / float64(numTokens-2)
		}
	}

	maxNgramRepeat := 0.0
	if numTokens >= 5 {
		ngrams4 := make(map[string]int)
		for i := 0; i < numTokens-3; i++ {
			key := tokens[i] + "|" + tokens[i+1] + "|" + tokens[i+2] + "|" + tokens[i+3]
			ngrams4[key]++
		}
		if len(ngrams4) > 1 {
			maxCount := 0
			for _, count := range ngrams4 {
				if count > maxCount {
					maxCount = count
				}
			}
			maxNgramRepeat = float64(maxCount) / float64(numTokens-3)
		}
	}

	stopCount := 0
	for _, t := range tokens {
		if fe.stopWords[t] {
			stopCount++
		}
	}
	stopWordRatio := float64(stopCount) / float64(numTokens)

	alphaCount := 0
	alphaWordRegex := regexp.MustCompile(`[\w\p{Han}]`)
	for _, t := range tokens {
		if alphaWordRegex.MatchString(t) {
			alphaCount++
		}
	}
	alphaRatio := float64(alphaCount) / float64(numTokens)

	totalTokenLen := 0
	for _, t := range tokens {
		totalTokenLen += len(t)
	}
	avgTokenLen := (float64(totalTokenLen) / float64(numTokens)) / 10.0

	resetFlag := 0.0
	for _, t := range tokens {
		if fe.resetKeywords[t] {
			resetFlag = 1.0
			break
		}
	}

	loopKeywordFlag := 0.0
	for _, t := range tokens {
		if fe.loopKeywords[t] {
			loopKeywordFlag = 1.0
			break
		}
	}

	puncChars := puncRegex.FindAllString(text, -1)
	puncDensity := 0.0
	if len(text) > 0 {
		puncDensity = float64(len(puncChars)) / float64(len(text))
	}

	textLower := strings.ToLower(text)
	vowels := len(vowelRegex.FindAllString(textLower, -1))
	alphas := len(alphaRegex.FindAllString(textLower, -1))
	vowelRatio := 0.0
	if alphas > 0 {
		vowelRatio = float64(vowels) / float64(alphas)
	}

	normalPatternCount := 0
	for _, t := range tokens {
		if fe.normalPatterns[t] {
			normalPatternCount++
		}
	}
	normalPatternDensity := float64(normalPatternCount) / float64(numTokens)

	textNoSpace := strings.ReplaceAll(textLower, " ", "")
	charCounts := make(map[rune]int)
	for _, c := range textNoSpace {
		charCounts[c]++
	}
	charRepetition := 0.0
	if len(textNoSpace) > 0 {
		charRepetition = float64(len(textNoSpace)-len(charCounts)) / float64(len(textNoSpace))
	}

	wordSaladScore := 0.0
	if numTokens > 3 {
		wordSaladScore = uniqueRatio * (1.0 - stopWordRatio)
	}

	tokenDiversity := entropy / (math.Log2(float64(numTokens)) + 1e-9)

	entropy2gram := 0.0
	if numTokens >= 2 {
		var ngrams2 []string
		for i := 0; i < numTokens-1; i++ {
			ngrams2 = append(ngrams2, tokens[i]+"|"+tokens[i+1])
		}
		entropy2gram = fe.calculateEntropy(ngrams2)
	}

	adversarialScore := 0.0
	if (resetFlag > 0 || loopKeywordFlag > 0) && normalPatternDensity < 0.1 {
		adversarialScore = 1.0
	}

	structLoopFlag := 0.0
	if numTokens >= 4 {
		for _, period := range []int{1, 2, 3, 4} {
			if numTokens >= period*2 {
				matches := 0
				for i := 0; i < numTokens-period; i++ {
					if tokens[i] == tokens[i+period] {
						matches++
					}
				}
				if float64(matches)/float64(numTokens-period) > 0.7 {
					structLoopFlag = 1.0
					break
				}
			}
		}
	}

	if numTokens == 1 {
		token := tokens[0]
		if len(token) > 10 {
			for _, period := range []int{1, 2, 3, 4} {
				if len(token) > period {
					matches := 0
					for i := 0; i < len(token)-period; i++ {
						if token[i] == token[i+period] {
							matches++
						}
					}
					if float64(matches)/float64(len(token)-period) > 0.8 {
						structLoopFlag = 1.0
						repetitionScore = 1.0
						break
					}
				}
			}
		}
	}

	semanticCoherence := float64(stopCount+normalPatternCount) / float64(numTokens)

	saladDiff := 0.0
	if numTokens > 6 {
		mid := numTokens / 2
		t1 := tokens[:mid]
		t2 := tokens[mid:]

		u1Tokens := make(map[string]bool)
		for _, t := range t1 {
			u1Tokens[t] = true
		}
		u1 := float64(len(u1Tokens)) / float64(len(t1))

		u2Tokens := make(map[string]bool)
		for _, t := range t2 {
			u2Tokens[t] = true
		}
		u2 := float64(len(u2Tokens)) / float64(len(t2))

		s1Count := 0
		for _, t := range t1 {
			if fe.stopWords[t] {
				s1Count++
			}
		}
		s1 := float64(s1Count) / float64(len(t1))

		s2Count := 0
		for _, t := range t2 {
			if fe.stopWords[t] {
				s2Count++
			}
		}
		s2 := float64(s2Count) / float64(len(t2))

		salad1 := u1 * (1.0 - s1)
		salad2 := u2 * (1.0 - s2)
		saladDiff = math.Abs(salad1 - salad2)
	}

	digits := len(digitRegex.FindAllString(text, -1))
	digitDensity := 0.0
	if len(text) > 0 {
		digitDensity = float64(digits) / float64(len(text))
	}

	features := map[string]float64{
		"length": length, "unique_ratio": uniqueRatio, "entropy": entropy,
		"repetition_score": repetitionScore, "max_ngram_repeat": maxNgramRepeat,
		"stop_word_ratio": stopWordRatio, "alpha_ratio": alphaRatio,
		"avg_token_len": avgTokenLen, "reset_flag": resetFlag,
		"loop_keyword_flag": loopKeywordFlag, "punc_density": puncDensity,
		"vowel_ratio": vowelRatio, "normal_pattern_density": normalPatternDensity,
		"char_repetition": charRepetition, "word_salad_score": wordSaladScore,
		"token_diversity": tokenDiversity, "entropy_2gram": entropy2gram,
		"adversarial_score": adversarialScore, "struct_loop_flag": structLoopFlag,
		"semantic_coherence": semanticCoherence, "salad_diff": saladDiff, "digit_density": digitDensity,
		"burstiness": burstiness, "keyword_persistence": keywordPersistence,
		"num_prompt_keywords": float64(numPromptKeywords),
	}

	vector := []float64{
		length, uniqueRatio, entropy, repetitionScore, maxNgramRepeat,
		stopWordRatio, alphaRatio, avgTokenLen, resetFlag, loopKeywordFlag,
		puncDensity, vowelRatio, normalPatternDensity, charRepetition, wordSaladScore,
		tokenDiversity, entropy2gram, adversarialScore, structLoopFlag, semanticCoherence,
		saladDiff, digitDensity, burstiness, keywordPersistence, float64(numPromptKeywords),
	}

	return &FeatureResult{Vector: vector, Features: features}
}
