import torch
import os
import hashlib
import functools
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pos_to_valence import get_valence_tensors
from flair.models import SequenceTagger

DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]

# sanity-check helper

def verify_needles(needles, tokenizer):
    bad = []
    for n in needles:
        if "full" not in n:
            bad.append(f"  MISSING 'full' key: {n}")
        toks = tokenizer.tokenize(n["answer"])
        if len(toks) != 1:
            bad.append(f"  answer '{n['answer']}' → {len(toks)} tokens: {toks}")
    if bad:
        print("[needles] WARNING – problematic needles found:")
        for b in bad:
            print(b)
    else:
        print(f"[needles] All {len(needles)} needles OK.")

# cloze sequence builder (tested but not used in the paper)

def build_cloze_from_parts(bg_ids, bg_val, needle, depth, tokenizer, max_len):
    n_full   = needle["full"]
    n_stem   = needle["stem"]
    n_answer = needle["answer"]

    needle_ids = tokenizer.encode(n_full, add_special_tokens=False)
    query_ids  = tokenizer.encode(n_stem, add_special_tokens=False)
    answer_ids = tokenizer.encode(n_answer, add_special_tokens=False)

    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102

    T          = len(bg_ids)
    inject_pos = max(0, min(int(T * depth), T))

    full_ids_untruncated = (
        [cls_id]
        + list(bg_ids[:inject_pos])
        + needle_ids
        + list(bg_ids[inject_pos:])
        + [sep_id]
        + query_ids
        + answer_ids
        + [sep_id]
    )

    answer_start_abs = (
        1
        + inject_pos
        + len(needle_ids)
        + (T - inject_pos)
        + 1
        + len(query_ids)
    )

    full_ids = full_ids_untruncated[:max_len]

    full_val = (
        [0.0]
        + list(bg_val[:inject_pos])
        + [1.0] * len(needle_ids)
        + list(bg_val[inject_pos:])
        + [0.0]
        + [0.5] * len(query_ids)
        + [1.0] * len(answer_ids)
        + [0.0]
    )[:max_len]

    labels = [-100] * len(full_ids)
    n_set  = 0
    for offset, aid in enumerate(answer_ids):
        pos = answer_start_abs + offset
        if pos < len(full_ids):
            labels[pos] = aid
            n_set += 1

    if n_set == 0:
        last = len(full_ids) - 2
        if last >= 0:
            labels[last] = answer_ids[0]

    attn_mask = [1] * len(full_ids)

    pad_len   = max_len - len(full_ids)
    pad_id    = tokenizer.pad_token_id or 0
    full_ids  = full_ids  + [pad_id] * pad_len
    labels    = labels    + [-100]   * pad_len
    attn_mask = attn_mask + [0]      * pad_len
    full_val  = full_val  + [0.0]    * pad_len

    return {
        "input_ids": full_ids,
        "attention_mask": attn_mask,
        "labels": labels,
        "valences": full_val,
        "depth": depth,
    }

# fast sentence boundary detection

@functools.lru_cache(maxsize=8)
def _build_period_token_set(tokenizer_name_or_path: str) -> frozenset:
    """
    Scan the vocabulary once and return the set of token IDs whose string
    representation contains a period character.

    Result is cached per tokenizer so the scan only happens once per process,
    regardless of how many times _find_sentence_boundaries is called.
    """
    tok   = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    vocab = tok.get_vocab()                      # {token_str: id}
    ids   = {tid for tok_str, tid in vocab.items() if "." in tok_str}
    print(f"[sentence_boundaries] period-token set built: "
          f"{len(ids)} tokens contain '.' in vocab of {len(vocab)}")
    return frozenset(ids)


def _find_sentence_boundaries(ids, tokenizer):
    """
    Return list of (start, end) token index pairs, one per sentence.
    Sentences are split on tokens whose string representation contains '.'.

    Uses a pre-built frozenset for O(1) lookup instead of calling
    tokenizer.decode() once per token (which was the startup bottleneck).
    """
    period_ids = _build_period_token_set(tokenizer.name_or_path)
    ids_list   = ids if isinstance(ids, list) else ids.tolist()

    boundaries = [0]
    for i in range(1, len(ids_list)):
        if ids_list[i - 1] in period_ids:
            boundaries.append(i)
    if boundaries[-1] != len(ids_list):
        boundaries.append(len(ids_list))
    return list(zip(boundaries[:-1], boundaries[1:]))


# NNP repeat-pair finder

MIN_REPEAT_DIST = 64


def _find_nnp_repeat_pair(ids, valences_list, tokenizer, min_dist=MIN_REPEAT_DIST):
    """
    Find a (first_pos, repeat_pos, sign) where:
      - the token has |valence| >= 0.75  (NNP/NNPS after 0.8 POS cap)
      - the same token id appeared >= min_dist positions earlier
    Returns None if no valid pair exists.
    When multiple pairs exist, picks one randomly.
    """
    ids_list = ids if isinstance(ids, list) else ids.tolist()

    first_seen = {}
    candidates = []
    for t, (tid, val) in enumerate(zip(ids_list, valences_list)):
        if abs(val) >= 0.75:
            if tid in first_seen:
                first_t = first_seen[tid]
                if t - first_t >= min_dist:
                    candidates.append((first_t, t, 1.0 if val > 0 else -1.0))
                    first_seen[tid] = t  # Bug 2/4 fix: chain to next gap
            else:
                first_seen[tid] = t

    if not candidates:
        return None
    return random.choice(candidates)


# plain AR example builder

def build_text_example(text, tokenizer, max_len, tagger,
                       upgrade_sentence_position=None, sample_idx=0):
    """
    Build a plain AR training example.

    If upgrade_sentence_position is not None, finds a proper noun repeat pair
    and boosts both sentences to ±1.0.  Falls back to boosting one cycling-
    depth sentence if no pair is found.
    """
    enc = tokenizer(str(text), truncation=True, max_length=max_len, padding=False)
    truncated_text = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
    _, valences = get_valence_tensors(truncated_text, tokenizer=tokenizer, tagger=tagger)

    ids    = enc["input_ids"]
    v_list = valences.tolist()
    if len(v_list) < len(ids):
        v_list += [0.1] * (len(ids) - len(v_list))
    final_v = torch.tensor(v_list[:len(ids)])

    if upgrade_sentence_position is not None:
        sentences = _find_sentence_boundaries(ids, tokenizer)
        pair = _find_nnp_repeat_pair(ids, v_list, tokenizer)
        if pair is not None:
            first_pos, repeat_pos, sign = pair
            for sel_start, sel_end in sentences:
                if sel_start <= first_pos < sel_end:
                    final_v[sel_start:sel_end] = sign
                if sel_start <= repeat_pos < sel_end:
                    final_v[sel_start:sel_end] = sign
        else:
            target_depth = DEPTHS[sample_idx % len(DEPTHS)]
            target_idx   = int(len(ids) * target_depth)
            starts    = [s for s, e in sentences]
            closest   = min(range(len(starts)), key=lambda i: abs(starts[i] - target_idx))
            sel_start, sel_end = sentences[closest]
            sign = 1.0 if (sample_idx % 2 == 0) else -1.0
            final_v[sel_start:sel_end] = sign

    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": ids[:],
        "valences": final_v,
    }

# datasets

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, tagger,
                 upgrade_sentence_position=None, cache_suffix=""):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.tagger    = tagger
        self.upgrade_sentence_position = upgrade_sentence_position

        sig        = hashlib.md5("".join(texts[:10]).encode()).hexdigest()[:8]
        fname      = f"ar_cache_{sig}_len{max_len}_15000_{cache_suffix}.pt"
        cache_path = os.path.join(os.path.dirname(__file__), fname)

        if os.path.exists(cache_path):
            print(f"Loading cached AR dataset: {cache_path}")
            self.cache = torch.load(cache_path)
        else:
            print(f"Pre-computing {len(texts)} AR examples…")
            self.cache = [
                build_text_example(str(t), tokenizer, max_len, tagger,
                                   upgrade_sentence_position, i)
                for i, t in enumerate(texts)
            ]
            torch.save(self.cache, cache_path)

    def __len__(self): return len(self.cache)
    def __getitem__(self, idx): return self.cache[idx]


class RetrievalDataset(Dataset):
    """
    Augmented retrieval dataset built entirely from already-cached TextDataset
    examples.  No Flair POS tagging or tokenisation — just Python list ops.

    Augmentations (zero additional Flair cost):
      1. All valid NNP pairs per window (not one random pair).
      2. Sliding 256-token subwindows of each 512-token document.

    Speed fix vs previous version:
      _find_sentence_boundaries is called ONCE per window, not once per pair.
      The boundary list is then reused for every pair found in that window.
    """
    WINDOW_SIZE   = 256
    WINDOW_STRIDE = 128

    def __init__(self, base_dataset, tokenizer, cache_suffix=""):
        self.tokenizer = tokenizer

        # Cache key
        first_ids = base_dataset[0]["input_ids"]
        if isinstance(first_ids, torch.Tensor):
            first_ids = first_ids[:8].tolist()
        sig        = hashlib.md5(
            (str(len(base_dataset)) + str(first_ids)).encode()
        ).hexdigest()[:8]
        cache_path = os.path.join(
            os.path.dirname(__file__),
            f"ret_aug_cache_{sig}{cache_suffix}.pt",
        )

        if os.path.exists(cache_path):
            print(f"Loading cached retrieval dataset: {cache_path}")
            self.examples = torch.load(cache_path)
            print(f"  {len(self.examples)} examples.")
            return

        print(f"Building augmented retrieval dataset "
              f"from {len(base_dataset)} documents…")
        self.examples = []

        for doc_idx in range(len(base_dataset)):
            ex        = base_dataset[doc_idx]
            ids_full  = ex["input_ids"]
            vals_full = ex["valences"]
            if isinstance(ids_full, torch.Tensor): ids_full  = ids_full.tolist()
            if isinstance(vals_full, torch.Tensor): vals_full = vals_full.tolist()

            attn = ex["attention_mask"]
            real_len = int(sum(
                v.item() if hasattr(v, "item") else v for v in attn))

            # Full sequence + sliding subwindows
            windows = [(0, real_len)]
            start = 0
            while start + self.WINDOW_SIZE <= real_len:
                windows.append((start, start + self.WINDOW_SIZE))
                start += self.WINDOW_STRIDE

            for w_start, w_end in windows:
                ids_w  = ids_full[w_start:w_end]
                vals_w = vals_full[w_start:w_end]

                # ── sentence boundaries: ONE call per window ──────────────
                sents = _find_sentence_boundaries(ids_w, tokenizer)

                # ── find all valid NNP pairs in this window ───────────────
                first_seen = {}
                for t, (tid, val) in enumerate(zip(ids_w, vals_w)):
                    if abs(val) >= 0.75:
                        if tid in first_seen:
                            first_t = first_seen[tid]
                            if t - first_t >= MIN_REPEAT_DIST:
                                sign      = 1.0 if val > 0 else -1.0
                                first_pos = first_t
                                repeat_pos = t

                                # Inject ±1.0 on both sentences — reuse sents
                                new_v = list(vals_w)
                                for s_start, s_end in sents:
                                    if (s_start <= first_pos  < s_end or
                                            s_start <= repeat_pos < s_end):
                                        for i in range(s_start, s_end):
                                            new_v[i] = sign

                                labels = [-100] * len(ids_w)
                                labels[repeat_pos] = ids_w[repeat_pos]

                                self.examples.append({
                                    "input_ids": ids_w,
                                    "attention_mask": [1] * len(ids_w),
                                    "labels": labels,
                                    "valences": new_v,
                                })
                                first_seen[tid] = t  # Bug 2 fix: chain pairs
                        else:
                            first_seen[tid] = t

        torch.save(self.examples, cache_path)
        n = len(self.examples)
        print(f"  {n} retrieval examples built and cached → {cache_path}")
        print(f"  avg {n / max(len(base_dataset), 1):.1f} examples per document")

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

class ClozeDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, tagger, needles,
                 depths=None, cache_suffix=""):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.tagger    = tagger
        self.needles   = needles
        self.depths    = depths if depths is not None else DEPTHS

        sig           = hashlib.md5("".join(texts[:5]).encode()).hexdigest()[:8]
        bg_cache_file = os.path.join(os.path.dirname(__file__),
                                     f"bg_cache_{sig}_{max_len}.pt")

        if os.path.exists(bg_cache_file):
            print(f"Loading cached backgrounds: {bg_cache_file}")
            self.bg_processed = torch.load(bg_cache_file)
        else:
            print(f"Pre-processing {len(texts)} backgrounds (POS tagging)…")
            max_needle_len = max(
                len(tokenizer.encode(n["full"], add_special_tokens=False))
                for n in needles)
            max_query_len = max(
                len(tokenizer.encode(n["stem"], add_special_tokens=False))
                for n in needles)
            max_answer_len = max(
                len(tokenizer.encode(n["answer"], add_special_tokens=False))
                for n in needles)
            overhead  = 3 + max_needle_len + max_query_len + max_answer_len
            bg_budget = max(max_len - overhead, 64)

            self.bg_processed = []
            for t in texts:
                enc = tokenizer(str(t), truncation=True,
                                max_length=bg_budget + 2, padding=False)
                trunc_text = tokenizer.decode(enc["input_ids"],
                                              skip_special_tokens=True)
                _, bg_val = get_valence_tensors(trunc_text, tokenizer, tagger)
                ids  = enc["input_ids"][1:-1]
                vals = bg_val[1 : len(ids) + 1].tolist()
                if len(vals) < len(ids):
                    vals += [0.1] * (len(ids) - len(vals))
                self.bg_processed.append({"ids": ids, "val": vals[:len(ids)]})

            torch.save(self.bg_processed, bg_cache_file)
            print(f"Background cache saved → {bg_cache_file}")

        self.index = [
            (bg_idx, needle, depth)
            for bg_idx in range(len(self.bg_processed))
            for needle in needles
            for depth  in self.depths
        ]
        print(f"ClozeDataset ready: {len(self.index)} examples "
              f"({len(self.bg_processed)} docs × {len(needles)} needles "
              f"× {len(self.depths)} depths)")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        bg_idx, needle, depth = self.index[idx]
        bg = self.bg_processed[bg_idx]
        return build_cloze_from_parts(
            bg["ids"], bg["val"], needle, depth,
            self.tokenizer, self.max_len,
        )


# collator

class CausalLMCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        max_len = max(len(e["input_ids"]) for e in examples)
        batch_ids, batch_mask, batch_labels, batch_valences = [], [], [], []

        for ex in examples:
            ids  = list(ex["input_ids"])
            mask = list(ex["attention_mask"])
            lbls = list(ex["labels"])
            vals = list(ex["valences"]) if not isinstance(ex["valences"], list) \
                   else ex["valences"]
            pad  = max_len - len(ids)

            batch_ids.append(   ids  + [self.pad_token_id] * pad)
            batch_mask.append(  mask + [0]                 * pad)
            batch_labels.append(lbls + [-100]              * pad)
            batch_valences.append(
                torch.tensor(vals + [0.0] * pad, dtype=torch.float)
            )

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.bool),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "valences": torch.stack(batch_valences),
        }

# high-level task helper

class LanguageTask:
    def __init__(self, model_name="bert-base-uncased", max_len=512,
                 tagger=None, upgrade_sentence_position=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_len)
        self.max_len                   = max_len
        self.tagger                    = tagger or SequenceTagger.load("flair/pos-english")
        self.upgrade_sentence_position = upgrade_sentence_position

    def _collator(self):
        return CausalLMCollator(
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        )

    def get_dataloader(self, texts, batch_size=32, num_workers=4,
                       shuffle=True, cache_suffix=""):
        ds = TextDataset(texts, self.tokenizer, self.max_len, self.tagger,
                         self.upgrade_sentence_position, cache_suffix)
        return DataLoader(ds, batch_size=batch_size, collate_fn=self._collator(),
                          shuffle=shuffle, num_workers=num_workers)

    def get_cloze_dataloader(self, texts, needles, depths=None,
                              batch_size=32, num_workers=4,
                              shuffle=True, cache_suffix=""):
        ds = ClozeDataset(texts, self.tokenizer, self.max_len, self.tagger,
                          needles, depths=depths, cache_suffix=cache_suffix)
        return DataLoader(ds, batch_size=batch_size, collate_fn=self._collator(),
                          shuffle=shuffle, num_workers=num_workers)