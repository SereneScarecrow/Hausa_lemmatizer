from hausa_lemmatizer import HausaLemmatizer

def main():
    # Инициализация лемматизатора
    lemmatizer = HausaLemmatizer()
    
    # Тестовые предложения на языке хауса
    test_sentences = [
        "Shin matsalar dabanci ta gari hukumomi ne a Kano?",
        "Mutane suna karatu a makaranta.",
        "Yara suna wasa a filin wasa.",
        "Malamai suna koyar da ilimi.",
        "Abinci yana dadi sosai."
    ]
    
    print("=== ДЕМОНСТРАЦИЯ РАБОТЫ ЛЕММАТИЗАТОРА ХАУСА ===\n")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Пример {i}:")
        print(f"Исходное предложение: {sentence}")
        
        # Анализ с раздельными токенами
        print("\n1. Анализ с раздельными токенами:")
        separate_analysis = lemmatizer.analyze_sentence_separate(sentence)
        for item in separate_analysis:
            print(f"   {item['word']:15} [{item['POS']:8}] -> {item['lemma']}")
        
        # Анализ с объединенными токенами
        print("\n2. Анализ с объединенными токенами:")
        combined_analysis = lemmatizer.analyze_sentence_combined(sentence)
        for item in combined_analysis:
            print(f"   {item['word']:15} [{item['POS']:8}] -> {item['lemma']}")
        
        # Различные форматы вывода
        print("\n3. Форматы вывода:")
        print(f"   Формат underscore: {lemmatizer.get_lemma_sentence(sentence, 'underscore')}")
        print(f"   Только леммы: {lemmatizer.get_lemma_sentence(sentence, 'lemma_only')}")
        print(f"   Объединенные: {lemmatizer.get_lemma_sentence(sentence, 'combined', combine_tokens=True)}")
        
        print("\n" + "="*60 + "\n")