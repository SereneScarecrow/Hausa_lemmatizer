import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class HausaLemmatizer:
    def __init__(self, plural_dict_path: Optional[str] = "plural_nouns.json", verb_dict_path: Optional[str] = None):
        """
        Initialize the Hausa lemmatizer with dictionaries.
        
        Args:
            plural_dict_path: Path to plural nouns dictionary (JSON)
            verb_dict_path: Path to verbs dictionary (JSON) - currently not implemented
        """
        # Load dictionary
        self.plural_dict = self._load_dictionary(plural_dict_path)
        
        # Define plural suffixes for rule-based processing
        # self.plural_suffixes = ['-', '']

    def _load_dictionary(self, dict_path: Optional[str]) -> Dict[str, str]:
        """
        Load a dictionary from JSON file.
        
        Args:
            dict_path: Path to dictionary file
            
        Returns:
            Dictionary or empty dict if file doesn't exist
        """
        if dict_path and Path(dict_path).exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading dictionary {dict_path}: {e}")
                return {}
        return {}

    def process_text(self, pos_tagged_text: List[str]) -> List[str]:
        """
        Process POS-tagged text to extract lemmas.
        
        Args:
            pos_tagged_text: List of tokens in 'word_POS' or 'part1+part2_POS1+POS2' format
            
        Returns:
            List of processed tokens with lemmas
        """
        processed_tokens = []
        
        for token in pos_tagged_text:
            # Split token into word and POS tag parts
            parts = token.split('_')
            if len(parts) != 2:
                processed_tokens.append(token)
                continue
                
            word_part, pos_part = parts
            
            # Handle different POS tag combinations
            if pos_part in ['NOUN+PART', 'NOUN+PRON']:
                # Remove possessive particle and sometimes plural suffix
                word = word_part.split('+')[0]
                lemma = self.process_word_by_pos(word, 'NOUN')
            elif pos_part == 'VERB+NOUN':
                # Process verbal nouns as nouns
                word = ''.join(word_part.split('+'))
                lemma = self.process_word_by_pos(word, 'NOUN')
            elif 'NUM' in pos_part:
                # Process numerals
                word = ''.join(word_part.split('+'))
                lemma = self.process_word_by_pos(word, 'NUM')
            else:
                # For everything else, use the first POS tag
                word = ''.join(word_part.split('+'))
                pos_tag = pos_part.split('+')[0]
                lemma = self.process_word_by_pos(word, pos_tag)
            
            processed_tokens.append(lemma)
        
        return processed_tokens

    def process_word_by_pos(self, word: str, pos_tag: str) -> str:
        """
        Process a word based on its part-of-speech tag.
        
        Args:
            word: The word to process
            pos_tag: POS tag of the word
            
        Returns:
            Lemma of the word
        """
        pos_handlers = {
            'PRON': self._process_pronoun,
            'PROPN': self._process_propn,
            'NUM': self._process_numeral,
            'NOUN': self._process_noun,
            'VERB': self._process_verb,
            'AUX': self._process_verb,
        }
        
        handler = pos_handlers.get(pos_tag, self._process_other)
        return handler(word, pos_tag)

    # Instead of the word the following functions return word tag
    # This is lemmatization requirement for specific task
    def _process_pronoun(self, word: str, pos_tag: str) -> str:
        """Process pronouns."""
        return "pron1"

    def _process_propn(self, word: str, pos_tag: str) -> str:
        """Process proper nouns."""
        return "propn1"

    def _process_numeral(self, word: str, pos_tag: str) -> str:
        """Process numerals."""
        return "num1"

    def _process_noun(self, word: str, pos_tag: str) -> str:
        """
        Process nouns: dictionary -> rules -> keep as is.
        
        Args:
            word: Noun to lemmatize
            pos_tag: POS tag (should be 'NOUN')
            
        Returns:
            Lemma of the noun
        """
        word_lower = word.lower()

        # Check if word is in plural dictionary
        if self.plural_dict and word_lower in self.plural_dict:
            print(f"'{word}' is plural, lemma: {self.plural_dict[word_lower]}")
            return self.plural_dict[word_lower]

        # Check if word is already singular (present in dictionary values)
        if self.plural_dict and word_lower in self.plural_dict.values():
            print(f"'{word}' is singular")
            return word_lower

        # Apply rule-based stemming
        # stem_by_rules = self._apply_noun_rules(word_lower)
        # if stem_by_rules != word_lower:
        #     return stem_by_rules

        return word_lower

    def _apply_noun_rules(self, word: str) -> str:
        """
        Apply rules to derive singular form from plural.
        
        Args:
            word: Noun to process
            
        Returns:
            Stemmed noun
        """
        for suffix in self.plural_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if stem:
                    return stem
        return word

    def _process_verb(self, word: str, pos_tag: str) -> str:
        """
        Process verbs: rules -> keep as is.
        
        Args:
            word: Verb to lemmatize
            pos_tag: POS tag ('VERB' or 'AUX')
            
        Returns:
            Lemma of the verb
        """
        word_lower = word.lower()

        # Check verb dictionary if available
        # if self.verb_dict and word_lower in self.verb_dict:
        #     return self.verb_dict[word_lower]

        # Apply rule-based stemming
        stem_by_rules = self._apply_verb_rules(word_lower)
        if stem_by_rules != word_lower:
            return stem_by_rules

        return word_lower

    def _apply_verb_rules(self, word: str) -> str:
        """
        Apply rules to derive verb base form.
        
        Args:
            word: Verb to process
            
        Returns:
            Stemmed verb
        """
        # Rule 1: endings 'ce', 'ci' -> 't'
        if word.endswith(('ce', 'ci')):
            word = word[:-2] + 't'
        
        # Rule 2: vowel after 's' -> 'sh'
        elif word[-1] in {'a', 'e', 'i', 'u'} and len(word) > 1 and word[-2] == 's':
            word = word[:-2] + 'sh'
        
        # Rule 3: remove final vowel
        elif word[-1] in {'a', 'e', 'i', 'u'}:
            word = word[:-1]
        
        return word

    def _process_other(self, word: str, pos_tag: str) -> str:
        """
        Process other POS tags (adjectives, adverbs, etc.).
        
        Args:
            word: Word to process
            pos_tag: POS tag
            
        Returns:
            Lowercased word
        """
        return word.lower()
