from datasets import load_dataset, Dataset
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pile_dataset")

print(f"Downloading Wikipedia (en) from monology/pile-uncopyrighted to {output_dir}...")

def filter_wikipedia():
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    count = 0
    for row in ds:
        meta = row.get('meta', {})
        if meta.get('pile_set_name') == 'Wikipedia (en)':
            yield row
            count += 1
            if count % 1000 == 0:
                print(f"Collected {count} articles...", end='\r')
            if count >= 100000:
                print("\nReached 100k articles limit.")
                break

# create dataset from generator
print("Starting download and filter process...")
filtered_ds = Dataset.from_generator(filter_wikipedia)

print(f"\nSaving {len(filtered_ds)} articles to disk at {output_dir}...")
filtered_ds.save_to_disk(output_dir)
print("Done!")
