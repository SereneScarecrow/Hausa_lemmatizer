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

    
    def _load_raw_dictionary(self, dict_path):
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
            return self.plural_dict[word_lower]
        
        if word_lower in self.plural_dict.values():
            return word_lower
        
        stem_by_rules = self._apply_noun_rules(word_lower)
        if stem_by_rules != word_lower:
            return stem_by_rules
        
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
        
        if self.verb_dict and word_lower in self.verb_dict:
            return self.verb_dict[word_lower]
        
        stem_by_rules = self._apply_verb_rules(word_lower)
        if stem_by_rules != word_lower:
            return stem_by_rules
        
        return word_lower
    
    def _apply_verb_rules(self, word):
        """Применяет правила для образования начальной формы глагола"""
        for suffix in self.verb_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if stem:
                    return stem
        return word
    
    def _process_other(self, word, pos_tag):
        return word.lower()
    
    def analyze_sentence(self, text):
        """
        Основная функция: возвращает список JSON-объектов с word, POS и lemma
        """
        tagged_tokens = self.pos_pipeline(text)
        result = []
        
        for token in tagged_tokens:
            word = token['word'].replace('▁', ' ').strip()
            if word:  # Пропускаем пустые токены
                pos_tag = token['entity_group']
                lemma = self.process_word_by_pos(word, pos_tag)
                
                result.append({
                    'word': word,
                    'POS': pos_tag,
                    'lemma': lemma
                })
        
        return result
    
    def get_lemma_sentence(self, text, format_type="underscore"):
        """
        Возвращает предложение в разных форматах из готового списка лемм
        
        format_type: 
          - "underscore": word_POS_lemma
          - "lemma_only": только леммы
          - "combined": объединенные слова с тегами
        """
        analysis = self.analyze_sentence(text)
        
        if format_type == "underscore":
            return " ".join([f"{item['word']}_{item['POS']}_{item['lemma']}" for item in analysis])
        
        elif format_type == "lemma_only":
            return " ".join([item['lemma'] for item in analysis])
        
        elif format_type == "combined":
            # Объединяем слова, которые были разделены токенизатором
            combined_tokens = self._combine_tokens(analysis, text)
            return " ".join([f"{item['word']}_{item['POS']}" for item in combined_tokens])
        
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
            if current['POS'] != 'PUNC' and i + 1 < len(tokens):
                combined_word = current['word']
                combined_pos = current['POS']
                j = i + 1
                
                while j < len(tokens):
                    next_token = tokens[j]
                    
                    # Прерываем если встретили пунктуацию
                    if next_token['POS'] == 'PUNC':
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
                        'lemma': self.process_word_by_pos(combined_word, combined_pos.split('+')[0])
                    })
                    i = j
                    continue
            
            # Если объединения не было, добавляем текущий токен
            result.append(current)
            i += 1
        
        return result
    
    def save_analysis(self, text, output_path):
        """Сохраняет анализ в JSON файл"""
        analysis = self.analyze_sentence(text)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    def print_detailed_analysis(self, text):
        """Печатает детальный анализ"""
        analysis = self.analyze_sentence(text)
        
        print(f"Исходный текст: {text}")
        print("\nТокенизированный анализ:")
        print("-" * 50)
        for item in analysis:
            print(f"{item['word']:15} [{item['POS']:8}] -> {item['lemma']}")
        print("-" * 50)
        
        print(f"\nФормат underscore: {self.get_lemma_sentence(text, 'underscore')}")
        print(f"Только леммы: {self.get_lemma_sentence(text, 'lemma_only')}")
        print(f"Объединенные слова: {self.get_lemma_sentence(text, 'combined')}")

# Пример использования
if __name__ == "__main__":
    lemmatizer = HausaLemmatizer()
    
    test_text = "Shin matsalar dabanci ta gari hukumomi ne a Kano?"
    
    # Получаем список JSON-объектов
    analysis_list = lemmatizer.analyze_sentence(test_text)
    print("Список JSON-объектов:")
    for item in analysis_list:
        print(json.dumps(item, ensure_ascii=False))
    
    print("\n" + "="*60)
    
    # Получаем предложения в разных форматах
    print("Предложение с тегами:", lemmatizer.get_lemma_sentence(test_text, "underscore"))
    print("Только леммы:", lemmatizer.get_lemma_sentence(test_text, "lemma_only"))
    print("Объединенные слова:", lemmatizer.get_lemma_sentence(test_text, "combined"))
    
    print("\n" + "="*60)
    
    # Детальный анализ
    lemmatizer.print_detailed_analysis(test_text)