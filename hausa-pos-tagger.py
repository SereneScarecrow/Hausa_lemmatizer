from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import pandas as pd
import string


class HausaPOSTagger:
    def __init__(self, model_name="masakhane/hausa-pos-tagger-afroxlmr"):
        """Initialize the Hausa POS tagger with the specified model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pos_pipeline = TokenClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer
        )

    def get_pos_text(self, text: str) -> list[dict]:
        """
        Returns a list of dictionaries with words and their part-of-speech tags.
        """
        # Get POS tags for each token
        tagged_tokens = self.pos_pipeline(text)
        tokens = []

        # Extract cleaned tokens
        for token in tagged_tokens:
            word = token['word'].replace('▁', ' ').strip()
            if word:
                tokens.append({
                    'word': word,
                    'pos': token['entity']
                })

        return tokens

    def _merge_consecutive_parts(self, word_parts: list[str], pos_tags: list[str]) -> tuple[list[str], list[str]]:
        """
        Merge consecutive word parts with identical POS tags.
        
        Args:
            word_parts: List of word fragments
            pos_tags: List of corresponding POS tags
            
        Returns:
            Tuple of (merged_words, merged_pos)
        """
        if not word_parts:
            return [], []

        merged_words = []
        merged_pos = []
        current_word = word_parts[0]
        current_pos = pos_tags[0]

        for i in range(1, len(word_parts)):
            if pos_tags[i] == current_pos:
                # Combine word parts with identical POS
                current_word += word_parts[i]
            else:
                # Save previous part and start new one
                merged_words.append(current_word)
                merged_pos.append(current_pos)
                current_word = word_parts[i]
                current_pos = pos_tags[i]

        # Add final part
        merged_words.append(current_word)
        merged_pos.append(current_pos)

        return merged_words, merged_pos

    def get_combined_pos_text(self, text: str) -> list[str]:
        """
        Combines split tokens and returns in 'word_POS' or 'part1+part2_POS1+POS2' format.
        Merges consecutive parts with identical POS tags.
        """
        # Get base tokens
        tokens = self.get_pos_text(text)

        # Words from original text
        text_words = text.split()
        result = []
        i = 0  # index in tokens
        j = 0  # index in text_words

        while i < len(tokens):
            if j < len(text_words):
                # Current word from text
                target_word = text_words[j]
                combined = tokens[i]['word']
                word_parts = [tokens[i]['word']]
                pos_tags = [tokens[i]['pos']]
                k = i

                # Try to combine following tokens
                while combined != target_word and k + 1 < len(tokens):
                    next_token = tokens[k + 1]
                    test_combined = combined + next_token['word']

                    if test_combined == target_word or target_word.startswith(test_combined):
                        combined = test_combined
                        word_parts.append(next_token['word'])
                        pos_tags.append(next_token['pos'])
                        k += 1
                    else:
                        break

                # If we assembled the complete word
                if combined == target_word:
                    # Merge consecutive identical parts
                    merged_words, merged_pos = self._merge_consecutive_parts(word_parts, pos_tags)

                    # Format the result
                    word_with_plus = "+".join(merged_words)
                    pos_with_plus = "+".join(merged_pos)
                    result.append(f"{word_with_plus}_{pos_with_plus}")
                    i = k + 1
                    j += 1
                else:
                    # Failed to combine
                    result.append(f"{tokens[i]['word']}_{tokens[i]['pos']}")
                    i += 1
            else:
                # Remaining tokens
                result.append(f"{tokens[i]['word']}_{tokens[i]['pos']}")
                i += 1

        return result

    def create_comparison_dataframe(self, text: str) -> pd.DataFrame:
        """
        Creates a DataFrame for comparing tokenization and POS tagging approaches.
        """
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation + '“”')
        text_without_punctuation = text.translate(translator)

        # Get combined POS tagging
        result = self.get_combined_pos_text(text_without_punctuation)
        combined_pos_result = " ".join(result)

        # Prepare data for DataFrame
        data = {
            'original_text': text_without_punctuation.split(),
            'combined_pos_format': combined_pos_result.split()
        }

        if len(data['original_text']) == len(data['combined_pos_format']):
            return pd.DataFrame(data)

        print("DataFrame wasn't created because only part of the text was analyzed")
        return data