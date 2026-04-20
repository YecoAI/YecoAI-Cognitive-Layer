import re
import math
from collections import Counter

class FeatureEngine:
    def __init__(self):
        self.stop_words = set([
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "it", "this", "that", "you", "i", "me", "my",
            "can", "could", "will", "would", "shall", "should", "do", "does", "did", "have", "has", "had", "be", "been", "being", "am", "are", "was", "were",
            "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "write", "explain", "describe", "summarize", "translate", "list", "show", "help", "please", "repetitive", "poem", "short", "story",
            "compose", "verse", "themed", "medical", "humor", "joke", "tell", "query", "response", "task", "instruction", "snippet", "basic", "casual", "generate", "nature",
            "numeric", "echo", "verse", "verse", "humor", "medical", "humour", "encouragement", "cheer", "repeated", "greeting", "casual", "instruction", "poetic", "repetition",
            "et", "in", "est", "non", "ad", "ut", "cum", "quod", "qui", "que", "sed", "si", "per", "ex", "de", "esse", "sunt",
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "也", "他", "一个", "说", "去", "谢谢", "我们", "你们", "他们",
            "di", "e", "il", "la", "che", "un", "una", "per", "con", "su", "mi", "ti", "si", "è", "ma", "ed", "se", "perché", "come", "molto", "sono", "ho", "ha", "abbiamo", "hanno"
        ])
        
        self.reset_keywords = set([
            "reset", "forget", "clear", "wipe", "purge", "override", "nullify", "reboot",
            "dimentica", "resetta", "cancella", "vuoto", "ignora", "pulisci",
            "重置", "忘记", "清除", "清空", "重启"
        ])

        self.loop_keywords = set([
            "loop", "ripeti", "repeat", "redundancy", "stuck", "bloccato", "endless",
            "循环", "重复", "卡住", "再次", "ancora", "infinito"
        ])
        
        self.normal_patterns = set([
            "hello", "hi", "thanks", "thank", "please", "could", "would", "how", "what", "can", "help", "need", "write", "explain", "describe", "tell", "show", "give", "make",
            "happy", "sad", "good", "bad", "better", "best", "great", "awesome", "cool", "nice",
            "mist", "roll", "silent", "peak", "forest", "breath", "morning", "light",
            "patient", "medical", "doctor", "glass", "opinion", "reply", "humor", "joke", "funny",
            "photosynthesis", "sunlight", "carbon", "dioxide", "oxygen", "energy", "sugar", "chlorophyll", "conversion",
            "def", "add", "return", "function", "parameter", "snippet", "code", "programming",
            "ready", "assist", "request", "help", "chat",
            "1984", "orwell", "george", "totalitarianism", "dystopian", "fiction", "novel", "surveillance", "aspect", "control",
            "capital", "france", "paris", "london", "rome", "berlin", "tokyo", "beijing", "washington", "city", "major", "hub",
            "Peak", "reach", "sky", "snow", "eagle", "fly", "nature",
            "salve", "ave", "gratias", "quomodo", "quis", "quid", "ubi", "quando", "ciao", "grazie", "per favore", "potresti", "spiega", "descrivi",
            "opera", "poesia", "storia", "scienza", "libro", "film", "musica", "viaggio", "cibo", "lavoro", "pomeriggio", "sera", "notte", "giorno",
            "你好", "谢谢", "请", "怎么", "什么", "哪里", "帮助", "解释", "描述", "分析", "研究", "讨论", "建议"
        ])

    def _tokenize(self, text):
        tokens = re.findall(r'\b\w+\b|[\u4e00-\u9fff]|[^\w\s]', text.lower())
        return [t[:-1] if t.endswith('s') and len(t) > 3 else t for t in tokens]

    def _calculate_entropy(self, items):
        if not items:
            return 0.0
        counts = Counter(items)
        total = len(items)
        probs = [count / total for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def _calculate_burstiness(self, tokens):
        if len(tokens) < 10:
            return 0.0
        
        positions = {}
        for i, token in enumerate(tokens):
            if token not in positions:
                positions[token] = []
            positions[token].append(i)
        
        intervals = []
        for token, pos_list in positions.items():
            if len(pos_list) > 1:
                for i in range(len(pos_list) - 1):
                    intervals.append(pos_list[i+1] - pos_list[i])
        
        if not intervals:
            return 0.0
            
        avg_int = sum(intervals) / len(intervals)
        variance = sum((x - avg_int) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance)
        
        cv = std_dev / avg_int if avg_int > 0 else 0.0
        return max(0.0, min(1.0, (cv - 1) / (cv + 1) if cv + 1 != 0 else 0.0))

    def _get_keywords(self, tokens):
        return [t for t in tokens if (len(t) >= 3 and t not in self.stop_words) or re.match(r'^\d+$', t)]

    def _get_common_prefix(self, s1, s2):
        if s1 in s2 or s2 in s1: return True
        min_len = min(len(s1), len(s2))
        if min_len < 3: return s1 == s2
        common = 0
        for i in range(min_len):
            if s1[i] == s2[i]: common += 1
            else: break
        return common / max(len(s1), len(s2)) > 0.6

    def extract_features(self, text, prompt=None):
        if not text:
            return [0.0] * 24, {
                "length": 0, "unique_ratio": 0.0, "entropy": 0.0, "repetition_score": 0.0, "max_ngram_repeat": 0.0,
                "stop_word_ratio": 0.0, "alpha_ratio": 0.0, "avg_token_len": 0.0, "reset_flag": 0.0, "loop_keyword_flag": 0.0, 
                "punc_density": 0.0, "vowel_ratio": 0.0, "normal_pattern_density": 0.0, "char_repetition": 0.0, "word_salad_score": 0.0, "token_diversity": 0.0,
                "entropy_2gram": 0.0, "adversarial_score": 0.0, "struct_loop_flag": 0.0, "semantic_coherence": 0.0,
                "salad_diff": 0.0, "digit_density": 0.0, "burstiness": 0.0, "keyword_persistence": 1.0
            }

        if re.match(r'^[01\s]{10,}$|^\d{15,}$', text.strip()):
            tokens = [text.strip()]
        else:
            tokens = self._tokenize(text)
        
        num_tokens = len(tokens)
        
        keyword_persistence = 1.0
        num_prompt_keywords = 0
        if prompt:
            prompt_tokens = self._tokenize(prompt)
            prompt_keywords = self._get_keywords(prompt_tokens)
            num_prompt_keywords = len(prompt_keywords)
            if prompt_keywords:
                text_tokens_set = set(tokens)
                matches = 0
                for pk in prompt_keywords:
                    if pk in text_tokens_set:
                        matches += 1
                    else:
                        if any(self._get_common_prefix(pk, tk) for tk in tokens if len(tk) >= 4):
                            matches += 1
                keyword_persistence = matches / num_prompt_keywords

        if num_tokens == 0:
            return [0.0] * 25, {
                "length": 0, "unique_ratio": 0.0, "entropy": 0.0, "repetition_score": 0.0, "max_ngram_repeat": 0.0,
                "stop_word_ratio": 0.0, "alpha_ratio": 0.0, "avg_token_len": 0.0, "reset_flag": 0.0, "loop_keyword_flag": 0.0, 
                "punc_density": 0.0, "vowel_ratio": 0.0, "normal_pattern_density": 0.0, "char_repetition": 0.0, "word_salad_score": 0.0, "token_diversity": 0.0,
                "entropy_2gram": 0.0, "adversarial_score": 0.0, "struct_loop_flag": 0.0, "semantic_coherence": 0.0,
                "salad_diff": 0.0, "digit_density": 0.0, "burstiness": 0.0, "keyword_persistence": keyword_persistence,
                "num_prompt_keywords": float(num_prompt_keywords)
            }

        length = math.log1p(num_tokens)
        unique_tokens = set(tokens)
        unique_ratio = len(unique_tokens) / num_tokens
        entropy = self._calculate_entropy(tokens)
        burstiness = self._calculate_burstiness(tokens)

        repetition_score = 0.0
        if num_tokens >= 3:
            ngrams = list(zip(tokens, tokens[1:], tokens[2:]))
            ngram_counts = Counter(ngrams)
            repeated_ngrams = sum(count for count in ngram_counts.values() if count > 1)
            repetition_score = repeated_ngrams / len(ngrams)
        
        max_ngram_repeat = 0.0
        if num_tokens >= 5:
             ngrams_4 = list(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))
             if len(ngrams_4) > 1:
                 max_ngram_repeat = Counter(ngrams_4).most_common(1)[0][1] / len(ngrams_4)

        stop_count = sum(1 for t in tokens if t in self.stop_words)
        stop_word_ratio = stop_count / num_tokens
        alpha_count = sum(1 for t in tokens if re.match(r'[\w\u4e00-\u9fff]', t))
        alpha_ratio = alpha_count / num_tokens
        avg_token_len = sum(len(t) for t in tokens) / num_tokens / 10.0
        reset_flag = 1.0 if any(t in self.reset_keywords for t in tokens) else 0.0
        loop_keyword_flag = 1.0 if any(t in self.loop_keywords for t in tokens) else 0.0

        punc_chars = re.findall(r'[^\w\s\u4e00-\u9fff]', text)
        punc_density = len(punc_chars) / len(text) if len(text) > 0 else 0.0

        text_lower = text.lower()
        vowels = len(re.findall(r'[aeiouàèìòù]', text_lower))
        alphas = len(re.findall(r'[a-zàèìòù]', text_lower))
        vowel_ratio = vowels / alphas if alphas > 0 else 0.0

        normal_pattern_count = sum(1 for t in tokens if t in self.normal_patterns)
        normal_pattern_density = normal_pattern_count / num_tokens
        
        char_counts = Counter(text_lower.replace(" ", ""))
        char_repetition = (len(text_lower.replace(" ", "")) - len(char_counts)) / len(text_lower.replace(" ", "")) if len(text_lower.replace(" ", "")) > 0 else 0.0
        word_salad_score = (unique_ratio * (1.0 - stop_word_ratio)) if num_tokens > 3 else 0.0
        token_diversity = entropy / (math.log2(num_tokens) + 1e-9)

        entropy_2gram = 0.0
        if num_tokens >= 2:
            ngrams_2 = list(zip(tokens, tokens[1:]))
            entropy_2gram = self._calculate_entropy(ngrams_2)

        adversarial_score = 1.0 if (reset_flag > 0 or loop_keyword_flag > 0) and normal_pattern_density < 0.1 else 0.0

        struct_loop_flag = 0.0
        if num_tokens >= 4:
            for period in [1, 2, 3, 4]:
                if num_tokens >= period * 2:
                    matches = 0
                    for i in range(num_tokens - period):
                        if tokens[i] == tokens[i+period]:
                            matches += 1
                    if matches / (num_tokens - period) > 0.7:
                        struct_loop_flag = 1.0
                        break
        
        if num_tokens == 1:
            token = tokens[0]
            if len(token) > 10:
                for period in [1, 2, 3, 4]:
                    matches = 0
                    for i in range(len(token) - period):
                        if token[i] == token[i+period]:
                            matches += 1
                    if matches / (len(token) - period) > 0.8:
                        struct_loop_flag = 1.0
                        repetition_score = 1.0  
                        break

        semantic_coherence = (stop_count + normal_pattern_count) / num_tokens

        salad_diff = 0.0
        if num_tokens > 6:
            mid = num_tokens // 2
            t1, t2 = tokens[:mid], tokens[mid:]
            u1, u2 = len(set(t1)) / len(t1), len(set(t2)) / len(t2)
            s1 = sum(1 for t in t1 if t in self.stop_words) / len(t1)
            s2 = sum(1 for t in t2 if t in self.stop_words) / len(t2)
            salad_diff = abs((u1 * (1.0 - s1)) - (u2 * (1.0 - s2)))

        digits = len(re.findall(r'\d', text))
        digit_density = digits / len(text) if len(text) > 0 else 0.0

        features_dict = {
            "length": float(length), "unique_ratio": float(unique_ratio), "entropy": float(entropy),
            "repetition_score": float(repetition_score), "max_ngram_repeat": float(max_ngram_repeat),
            "stop_word_ratio": float(stop_word_ratio), "alpha_ratio": float(alpha_ratio),
            "avg_token_len": float(avg_token_len), "reset_flag": float(reset_flag),
            "loop_keyword_flag": float(loop_keyword_flag), "punc_density": float(punc_density),
            "vowel_ratio": float(vowel_ratio), "normal_pattern_density": float(normal_pattern_density),
            "char_repetition": float(char_repetition), "word_salad_score": float(word_salad_score),
            "token_diversity": float(token_diversity), "entropy_2gram": float(entropy_2gram),
            "adversarial_score": float(adversarial_score), "struct_loop_flag": float(struct_loop_flag),
            "semantic_coherence": float(semantic_coherence), "salad_diff": float(salad_diff), "digit_density": float(digit_density),
            "burstiness": float(burstiness), "keyword_persistence": float(keyword_persistence),
            "num_prompt_keywords": float(num_prompt_keywords)
        }

        feature_vector = [
            length, unique_ratio, entropy, repetition_score, max_ngram_repeat,
            stop_word_ratio, alpha_ratio, avg_token_len, reset_flag, loop_keyword_flag,
            punc_density, vowel_ratio, normal_pattern_density, char_repetition, word_salad_score, 
            token_diversity, entropy_2gram, adversarial_score, struct_loop_flag, semantic_coherence,
            salad_diff, digit_density, burstiness, keyword_persistence, float(num_prompt_keywords)
        ]

        return feature_vector, features_dict
