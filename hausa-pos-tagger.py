from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import string
import pandas as pd

class HausaPOSTagger:
    def __init__(self, model_name="masakhane/hausa-pos-tagger-afroxlmr"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pos_pipeline = TokenClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer)

    def get_pos_text(lemmatizer, text: str) -> str:
        """
        Возвращает список словарей со словами и частями речи.
        """
        # Получаем POS-теги для каждого токена
        tagged_tokens = lemmatizer.pos_pipeline(text)
        tokens = []

        # Извлекаем очищенные токены
        for token in tagged_tokens:
            word = token['word'].replace('▁', ' ').strip()
            if word:
                tokens.append({
                    'word': word,
                    'pos': token['entity']
                })

        return tokens


    def get_combined_pos_text(lemmatizer, text: str) -> str:
        """
        Объединяет разделенные токены и возвращает в формате 'слово_POS' или 'часть1+часть2_POS1+POS2'.
        Объединяет подряд идущие одинаковые POS-теги и соответствующие части слов.
        """
        def merge_consecutive_parts(word_parts, pos_tags):
            """Объединяет подряд идущие части слов с одинаковыми POS-тегами"""
            if not word_parts:
                return [], []

            merged_words = []
            merged_pos = []
            current_word = word_parts[0]
            current_pos = pos_tags[0]

            for i in range(1, len(word_parts)):
                if pos_tags[i] == current_pos:
                    # Объединяем части слов с одинаковыми POS
                    current_word += word_parts[i]
                else:
                    # Сохраняем предыдущую часть и начинаем новую
                    merged_words.append(current_word)
                    merged_pos.append(current_pos)
                    current_word = word_parts[i]
                    current_pos = pos_tags[i]

            # Добавляем последнюю часть
            merged_words.append(current_word)
            merged_pos.append(current_pos)

            return merged_words, merged_pos

        # Получаем базовые токены
        tokens = HausaPOSTagger.get_pos_text(lemmatizer, text)

        # Слова из текста
        text_words = text.split()
        result = []
        i = 0  # индекс в tokens
        j = 0  # индекс в text_words

        while i < len(tokens):
            if j < len(text_words):
                # Текущее слово из текста
                target_word = text_words[j]
                combined = tokens[i]['word']
                word_parts = [tokens[i]['word']]
                pos_tags = [tokens[i]['pos']]
                k = i

                # Пытаемся объединить следующие токены
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

                # Если собрали полное слово
                if combined == target_word:
                    # Объединяем подряд идущие одинаковые части
                    merged_words, merged_pos = merge_consecutive_parts(word_parts, pos_tags)

                    # Формируем результат
                    word_with_plus = "+".join(merged_words)
                    pos_with_plus = "+".join(merged_pos)
                    result.append(f"{word_with_plus}_{pos_with_plus}")
                    i = k + 1
                    j += 1
                else:
                    # Не удалось объединить
                    result.append(f"{tokens[i]['word']}_{tokens[i]['pos']}")
                    i += 1
            else:
                # Остались лишние токены
                result.append(f"{tokens[i]['word']}_{tokens[i]['pos']}")
                i += 1

        return result, " ".join(result)

    def create_comparison_dataframe(self, lemmatizer, text: str) -> pd.DataFrame:
        """
        Создает DataFrame
        """
        translator = str.maketrans('', '', string.punctuation)
        text_without_punctuation = text.translate(translator)

        # 1. Получаем раздельную разметку
        separate_result = lemmatizer.get_pos_text(text_without_punctuation)

        # 2. Получаем объединенную разметку
        result, combined_pos_result = lemmatizer.get_combined_pos_text(text_without_punctuation)
        combined_lemma_result = [self.process_word_by_pos(*i.split('_')) for i in result]

        # 3. Подготавливаем данные для DataFrame
        data = {
            'original_text': text_without_punctuation.split(),
            # 'separate_format': separate_result.split(),
            'combined_pos_format': combined_pos_result.split(),
            'combined_lemma_format': combined_pos_result.split(),
        }

        df = pd.DataFrame(data)

        return df