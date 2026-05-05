import json
import os
import random
import tarfile
import torch
from typing import List
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm 

def _is_ascii(s: str) -> bool:
    try:
        s.encode('ascii')
        return True
    except Exception:
        return False

def get_cache_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "pile_dataset")

def load_pile_dataset(split='train', max_samples=1000, min_len=500,
                      train_ratio=0.9, seed=42):
    cache_dir  = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    full_cache = os.path.join(cache_dir, "pile_wikipedia_full_filtered.pt")

    if os.path.exists(full_cache):
        print(f"Loading full Wikipedia corpus from cache: {full_cache}")
        out = torch.load(full_cache)
        print(f"  {len(out)} articles in cache.")
    else:
        print("Streaming Wikipedia articles from Pile (this runs once)...")
        
        # stream the dataset
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        out = []
        
        # setup tqdm: total=100000 because that is our target reservoir size
        pbar = tqdm(total=100000, desc="Collecting Wikipedia Articles")
        
        for row in ds:
            meta = row.get("meta", {})
            if meta.get("pile_set_name") != "Wikipedia (en)":
                continue
            
            text = row.get("text", "").strip()
            # Filter by word count and ASCII
            if len(text.split()) >= min_len and _is_ascii(text):
                out.append(text)
                pbar.update(1) # Only update when we find a match
            
            if len(out) >= 100000:
                break
        
        pbar.close()
        torch.save(out, full_cache)
        print(f"  Saved {len(out)} articles to {full_cache}")

    # shuffle + split
    random.seed(seed)
    shuffled = list(out)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    if split == "train":
        subset = shuffled[:split_idx]
    else:
        subset = shuffled[split_idx:]

    return subset[:max_samples]


def load_american_stories_dataset(split='train', max_samples=1000, min_len=50, years: List[str] = None):
    if years is None:
        years = ['1960']

    if isinstance(years, (int, str)):
        years = [str(years)]
    else:
        years = [str(y) for y in years]

    out = []
    for year in years:
        try:
            filename = f"faro_{year}.tar.gz"
            file_path = hf_hub_download(repo_id="dell-research-harvard/AmericanStories", filename=filename, repo_type="dataset")
            
            with tarfile.open(file_path, "r:gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith('.json'):
                        continue
                    
                    f = tar.extractfile(member)
                    if f:
                        try:
                            content = f.read()
                            data = json.loads(content)
                            if 'full articles' in data and isinstance(data['full articles'], list):
                                for article_data in data['full articles']:
                                    text = article_data.get('article', '').strip()
                                    if len(text.split()) >= min_len and _is_ascii(text):
                                        out.append(text)
                                    
                                    if max_samples and len(out) >= max_samples:
                                        return out
                        except Exception:
                            continue
        except Exception as e:
            print(f"Error loading year {year}: {e}")
            continue
            
    return out


def load_longbench_dataset(split='train', max_samples=1000, min_len=50, max_len=None, **kwargs):
    try:
        ds = load_dataset('THUDM/LongBench-v2', split=split, streaming=True)
    except Exception as e:
        print(f"Error loading LongBench dataset: {e}")
        return []

    out = []
    for row in ds:
        text = row.get('context', '').strip()
        
        # Check constraints
        if len(text.split()) < min_len:
            continue
        if max_len is not None and len(text.split()) > max_len:
            continue
            
        if _is_ascii(text):
             out.append(text)
             
        if max_samples and len(out) >= max_samples:
            break
    return out


def load_task_dataset(split='train', max_samples=1000, min_len=50, dataset_name='pile', **kwargs):
    if dataset_name == 'pile':
        return load_pile_dataset(split=split, max_samples=max_samples, min_len=min_len)
    elif dataset_name == 'american_stories':
        return load_american_stories_dataset(split=split, max_samples=max_samples, min_len=min_len, **kwargs)
    elif dataset_name == 'longbench':
         return load_longbench_dataset(split=split, max_samples=max_samples, min_len=min_len, **kwargs)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")