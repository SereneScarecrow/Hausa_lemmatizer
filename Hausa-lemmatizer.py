import json
from pathlib import Path
import pandas as pd

class HausaLemmatizer:
    def __init__(self):

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