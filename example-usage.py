from hausa_pos_tagger import HausaPOSTagger
from hausa_lemmatizer import HausaLemmatizer

# Initialize
tagger = HausaPOSTagger()
lemmatizer = HausaLemmatizer("plural_nouns.json")

# Process text
text = "Yara suna zuwa makaranta."

# Get POS tags
pos_tags = tagger.get_combined_pos_text(text)
# Returns: ['Yara_NOUN', 'suna_VERB', 'zuwa_VERB', 'makaranta_NOUN']

# Get lemmas
lemmas = lemmatizer.process_text(pos_tags)
# Returns lemmatized forms