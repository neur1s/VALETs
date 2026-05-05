import torch
## need to install flair and transformers! and update pyarrow
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoTokenizer
from preprocess import load_task_dataset

def get_valence_tensors(text, tokenizer=None, tagger=None):
    if tagger is None:
        tagger = SequenceTagger.load("flair/pos-english")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # POS to valence mapping
    # arbitrarily, let's say we use positive for singular/present and negative for plural/past
    # POS to valence mapping (scaled by 0.8 to reserve headroom for injected needles)
    val_map = {
        # nouns: priority 1.0 -> 0.8
        "NN": 0.8, "NNP": 0.8, 
        "NNS": -0.8, "NNPS": -0.8, 
        
        # verbs: priority 0.8 -> 0.64
        "VB": 0.64, "VBP": 0.64, "VBZ": 0.64, 
        "VBD": -0.64, "VBN": -0.64, "VBG": -0.64, 
        
        # numbers & identifiers: priority 0.7 -> 0.56
        "CD": 0.56,   
        "ADD": -0.56,  
        
        # adjectives & affixes: priority 0.6 -> 0.48
        "JJ": 0.48, "JJR": 0.48, "JJS": -0.48,
        "AFX": -0.48, 
        
        # foreign words: priority 0.5 -> 0.4
        "FW": 0.4,
        
        # adverbs: priority 0.4 -> 0.32
        "RB": 0.32, "RBR": 0.32, "RBS": -0.32, "WRB": -0.32,
        
        # pronouns & modals: priority 0.3 -> 0.24
        "PRP": 0.24, "PRP$": 0.24, "WP": 0.24, "WP$": -0.24,
        "MD": -0.24,  
        "EX": -0.24,  
        
        # functional & structural: priority 0.1 -> 0.08
        "DT": 0.08,   
        "CC": 0.08,   
        "IN": 0.08, "TO": 0.08, "RP": -0.08, "PDT": -0.08, "WDT": -0.08,
        "POS": -0.08,  
        "UH": 0.08,   
        
        # non-semantic/formatting: remains 0.0
        "LS": 0.0, "SYM": 0.0, ".": 0.0, ",": 0.0, ":": 0.0, "(": 0.0, ")": 0.0, "''": 0.0, "``": 0.0
    }

    # flair POS tagging 
    sentence = Sentence(text)
    tagger.predict(sentence)
    
    # extract tags following the order of words
    word_tags = [token.get_label('pos').value for token in sentence] # get POS tag
    word_valences = [val_map.get(tag, 0.1) for tag in word_tags] # translate POS tag to valence level

    # subword tokenization
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    word_ids = encoding.word_ids() 

    # broadcasting valence to sub-word tokens 
    token_valences = []
    for w_idx in word_ids:
        if w_idx is None:
            token_valences.append(0.0) # padding or special tokens
        else:
            # Word id indexing ensures all sub-tokens of a word get the same valence
            # Clamp index to avoid potential out-of-bounds (rare tokenizer mismatches)
            safe_idx = min(w_idx, len(word_valences) - 1)
            token_valences.append(word_valences[safe_idx])
            
    return input_ids, torch.tensor(token_valences)

# example usage with Wikitext-style content
if __name__ == "__main__":
    texts = load_task_dataset(split='train', max_samples=1)
    sample_text = texts[0]
    print(f"Sample text: {sample_text}\n")
    ids, valences = get_valence_tensors(sample_text)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(ids[0])}")
    print(f"Valence Levels: {valences.tolist()}")