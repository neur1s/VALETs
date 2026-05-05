import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer
from language_task import ClozeDataset
from needles import make_procedural_needles
from flair.models import SequenceTagger

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tagger = SequenceTagger.load("flair/pos-english")
rng = random.Random(42)
needles = make_procedural_needles(tokenizer, 3, rng=rng)
texts = ["Some background text about history . " * 30]

ds = ClozeDataset(texts, tokenizer, 512, tagger, needles=needles)
ex = ds[0]
ids = ex["input_ids"]
decoded = tokenizer.convert_ids_to_tokens(ids)

# find [SEP] positions — should be two: end of bg, end of query
sep_positions = [i for i, t in enumerate(decoded) if t == '[SEP]']
print(f"[SEP] positions: {sep_positions}")
print(f"Sequence length: {len(ids)}")
print(f"Tokens around first [SEP]: {decoded[max(0,sep_positions[0]-5):sep_positions[0]+5]}")
print(f"Tokens after first [SEP]:  {decoded[sep_positions[0]:sep_positions[0]+15]}")
print(f"Label position 197 context: {decoded[190:205]}")
labels = ex["labels"]

# find label positions
label_positions = [i for i, l in enumerate(labels) if l != -100]
print(f"Label positions: {label_positions}")
print(f"Answer token:    '{tokenizer.convert_ids_to_tokens(labels[label_positions[0]])}'")
print(f"Expected:        '{needles[0]['answer']}'")

# confirm the needle appears in the input
decoded = tokenizer.decode(ids)
print(f"Needle in input: {needles[0]['full'] in decoded}")
print(f"Query in input:  {needles[0]['stem'] in decoded}")