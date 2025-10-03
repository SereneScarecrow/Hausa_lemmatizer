from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import json
from pathlib import Path

class HausaLemmatizer:
    def __init__(self, model_name="masakhane/hausa-pos-tagger-afroxlmr"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pos_pipeline = TokenClassificationPipeline(
            model=self.model, 
            tokenizer=self.tokenizer)

        # Загружаем словари
        self.plural_dict = self._load_dictionary("plural_nouns.json")
        # self.verb_dict = self._load_dictionary(verb_dict_path)

    
    def _load_dictionary(self, dict_path):
        """Загружает словарь без изменений"""
        if dict_path and Path(dict_path).exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки словаря {dict_path}: {e}")
                return {}
        return {}
    
    def process_word_by_pos(self, word, pos_tag):
        """Обрабатывает слово в зависимости от части речи"""
        
        if pos_tag == 'PRON':
            return self._process_pronoun(word, pos_tag)
        elif pos_tag == 'NUM':
            return self._process_numeral(word, pos_tag)
        elif pos_tag == 'NOUN':
            return self._process_noun(word, pos_tag)
        elif pos_tag in ['VERB', 'AUX']:
            return self._process_verb(word, pos_tag)
        else:
            return self._process_other(word, pos_tag)
    
    def _process_pronoun(self, word, pos_tag):
        return "pron"
    
    def _process_numeral(self, word, pos_tag):
        return "num"
    
    def _process_noun(self, word, pos_tag):
        """Обработка существительных: словарь -> правила -> как есть"""
        word_lower = word.lower()
        
        if self.plural_dict and word_lower in self.plural_dict:
            print('is plural')
            return self.plural_dict[word_lower]
        
        if word_lower in self.plural_dict.values():
            print('is singular')
            return word_lower
        
        # stem_by_rules = self._apply_noun_rules(word_lower)
        # if stem_by_rules != word_lower:
        #     return stem_by_rules
        
        return word_lower
    
    def _apply_noun_rules(self, word):
        """Применяет правила для образования единственного числа"""
        for suffix in self.plural_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if stem:
                    return stem
        return word
    
    def _process_verb(self, word, pos_tag):
        """Обработка глаголов: словарь -> правила -> как есть"""
        word_lower = word.lower()
        
        # if self.verb_dict and word_lower in self.verb_dict:
        #     return self.verb_dict[word_lower]
        
        # stem_by_rules = self._apply_verb_rules(word_lower)
        # if stem_by_rules != word_lower:
        #     return stem_by_rules
        
        return word_lower
    
    def _apply_verb_rules(self, word):
        """Применяет правила для образования начальной формы глагола"""
        # Правило 1: окончания 'ce', 'ci' -> 't'
        if word.endswith(('ce', 'ci')):
            word = word[:-2] + 't'
        
        # Правило 2: гласная после 's' -> 'sh'
        elif word[-1] in {'a', 'e', 'i', 'u'} and len(word) > 1 and word[-2] == 's':
            word = word[:-2] + 'sh'
        
        # Правило 3: удаление конечной гласной
        elif word[-1] in {'a', 'e', 'i', 'u'}:
            word = word[:-1]
        
        return word
    
    def _process_other(self, word, pos_tag):
        return word.lower()
    
    def analyze_sentence_separate(self, text):
        """
        Анализ с раздельными словами (без объединения токенов)
        Возвращает список JSON-объектов с word, POS и lemma
        """
        tagged_tokens = self.pos_pipeline(text)
        print(tagged_tokens)
        result = []
        
        for token in tagged_tokens:
            word = token['word'].replace('▁', ' ').strip()
            if word:  # Пропускаем пустые токены
                pos_tag = token['entity']
                lemma = self.process_word_by_pos(word, pos_tag)
                
                result.append({
                    'word': word,
                    'POS': pos_tag,
                    'lemma': lemma
                })
        
        return result
    
    def analyze_sentence_combined(self, text):
        """
        Анализ с объединенными словами (склеивает разделенные токены)
        Возвращает список JSON-объектов с word, POS и lemma
        """
        tagged_tokens = self.pos_pipeline(text)
        print(tagged_tokens)
        
        # Сначала получаем базовые токены
        base_tokens = []
        for token in tagged_tokens:
            word = token['word'].replace('▁', ' ').strip()
            if word:
                base_tokens.append({
                    'word': word,
                    'POS': token['entity'],
                    'lemma': None  # будет заполнено после объединения
                })
        
        # Объединяем токены
        combined_tokens = self._combine_tokens(base_tokens, text)
        
        # Теперь применяем лемматизацию к объединенным словам
        for token in combined_tokens:
            token['lemma'] = self.process_word_by_pos(token['word'], token['POS'].split('+')[0])
        
        return combined_tokens
    
    def get_lemma_sentence(self, text, format_type="underscore", combine_tokens=False):
        """
        Возвращает предложение в разных форматах
        
        format_type: 
          - "underscore": word_POS_lemma
          - "lemma_only": только леммы
          - "combined": объединенные слова с тегами
        
        combine_tokens: если True, использует объединенные токены
        """
        if combine_tokens:
            analysis = self.analyze_sentence_combined(text)
        else:
            analysis = self.analyze_sentence_separate(text)
        
        if format_type == "underscore":
            return " ".join([f"{item['word']}_{item['POS']}_{item['lemma']}" for item in analysis])
        
        elif format_type == "lemma_only":
            return " ".join([item['lemma'] for item in analysis])
        
        elif format_type == "combined":
            return " ".join([f"{item['word']}_{item['POS']}" for item in analysis])
        
        else:
            return " ".join([item['word'] for item in analysis])
    
    def _combine_tokens(self, tokens, original_text):
        """
        Объединяет токены, которые были разделены токенизатором
        """
        if not tokens:
            return []
        
        result = []
        i = 0
        original_lower = original_text.lower()
        
        while i < len(tokens):
            current = tokens[i]
            
            # Если это не пунктуация, пытаемся объединить с последующими
            if current['POS'] != 'PUNCT' and i + 1 < len(tokens):
                combined_word = current['word']
                combined_pos = current['POS']
                j = i + 1
                
                while j < len(tokens):
                    next_token = tokens[j]
                    
                    # Прерываем если встретили пунктуацию
                    if next_token['POS'] == 'PUNCT':
                        break
                    
                    test_combination = combined_word + next_token['word']
                    
                    # Проверяем есть ли комбинация в оригинальном тексте
                    if test_combination.lower() in original_lower:
                        combined_word = test_combination
                        combined_pos = f"{combined_pos}+{next_token['POS']}"
                        j += 1
                    else:
                        break
                
                if j > i + 1:  # Было объединение
                    result.append({
                        'word': combined_word,
                        'POS': combined_pos,
                        'lemma': None
                    })
                    i = j
                    continue
            
            # Если объединения не было, добавляем текущий токен
            result.append(current)
            i += 1
        
        return result
    
    def save_analysis(self, text, output_path, combine_tokens=False):
        """Сохраняет анализ в JSON файл"""
        if combine_tokens:
            analysis = self.analyze_sentence_combined(text)
        else:
            analysis = self.analyze_sentence_separate(text)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    def print_detailed_analysis(self, text, combine_tokens=False):
        """Печатает детальный анализ"""
        if combine_tokens:
            analysis = self.analyze_sentence_combined(text)
            mode = "с объединенными токенами"
        else:
            analysis = self.analyze_sentence_separate(text)
            mode = "с раздельными токенами"
        
        print(f"Исходный текст: {text}")
        print(f"Режим анализа: {mode}")
        print("\nТокенизированный анализ:")
        print("-" * 50)
        for item in analysis:
            print(f"{item['word']:15} [{item['POS']:8}] -> {item['lemma']}")
        print("-" * 50)
        
        print(f"\nФормат underscore: {self.get_lemma_sentence(text, 'underscore', combine_tokens)}")
        print(f"Только леммы: {self.get_lemma_sentence(text, 'lemma_only', combine_tokens)}")
        print(f"Объединенные слова: {self.get_lemma_sentence(text, 'combined', combine_tokens)}")

