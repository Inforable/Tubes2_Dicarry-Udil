import os
import re
import numpy as np
import json

class Tokenizer:
    def __init__(self, start_token="<start>", end_token="<end>", pad_token="<pad>"):
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0
        self.max_length = 0

    def clean_caption(self, caption):
        caption = caption.lower()
        caption = re.sub(r'[^a-z0-9\s]', '', caption)
        caption = f"{self.start_token} {caption} {self.end_token}"
        caption = re.sub(r'\s+', ' ', caption).strip()
        return caption

    def fit_on_texts(self, texts):
        words = set()
        for text in texts:
            words.update(text.split())
            self.max_length = max(self.max_length, len(text.split()))
        
        # Reserved tokens
        sorted_words = [self.pad_token, self.start_token, self.end_token] + sorted(list(words - {self.pad_token, self.start_token, self.end_token}))
        
        self.word_index = {word: i for i, word in enumerate(sorted_words)}
        self.index_word = {i: word for word, i in self.word_index.items()}
        self.vocab_size = len(sorted_words)

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index[word] for word in text.split() if word in self.word_index]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences, maxlen=None):
        if maxlen is None:
            maxlen = self.max_length
        
        padded_sequences = np.zeros((len(sequences), maxlen), dtype='int32')
        for i, seq in enumerate(sequences):
            if len(seq) > maxlen:
                padded_sequences[i] = seq[:maxlen]
            else:
                padded_sequences[i, :len(seq)] = seq
        return padded_sequences

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'word_index': self.word_index,
                'max_length': int(self.max_length)
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        obj = cls()
        obj.word_index = data['word_index']
        obj.index_word = {int(i): word for word, i in obj.word_index.items()}
        obj.vocab_size = len(obj.word_index)
        obj.max_length = data['max_length']
        return obj

def load_captions(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:] # Skip header
    
    mapping = {}
    for line in lines:
        parts = line.split(',', 1)
        if len(parts) < 2:
            continue
        img_id, caption = parts[0], parts[1].strip()
        if img_id not in mapping:
            mapping[img_id] = []
        mapping[img_id].append(caption)
    return mapping

def split_data(mapping, train_count=6000, val_count=1000, test_count=1000):
    all_images = sorted(list(mapping.keys()))
    
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    train_imgs = all_images[:train_count]
    val_imgs = all_images[train_count : train_count + val_count]
    test_imgs = all_images[train_count + val_count : train_count + val_count + test_count]
    
    return train_imgs, val_imgs, test_imgs

def main():
    captions_path = "data/flickr8k/captions.txt"
    save_dir = "data/flickr8k"
    
    # Check if captions exist
    if not os.path.exists(captions_path):
        print(f"Error: {captions_path} not found.")
        return

    # Load mapping
    mapping = load_captions(captions_path)
    print(f"Loaded {len(mapping)} images with captions.")

    # Split data (6000/1000/1000)
    train_imgs, val_imgs, test_imgs = split_data(mapping)
    print(f"Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    # Build vocabulary from training set
    tokenizer = Tokenizer()
    train_captions = []
    for img_id in train_imgs:
        for cap in mapping[img_id]:
            train_captions.append(tokenizer.clean_caption(cap))
    
    tokenizer.fit_on_texts(train_captions)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Max caption length: {tokenizer.max_length}")

    print("\n--- Running Preprocessing Sanity Check ---")
    sample_text = mapping[train_imgs[0]][0]
    cleaned = tokenizer.clean_caption(sample_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = tokenizer.pad_sequences(seq)
    
    print(f"Original Text : {sample_text}")
    print(f"Cleaned Text  : {cleaned}")
    print(f"Sequence IDs  : {seq[0]}")
    print(f"Padded Shape  : {padded.shape}")
    print(f"Padded Array  : {padded[0][:10]}... (truncated)")
    print("------------------------------------------\n")

    # Save tokenizer and split info
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }
    with open(os.path.join(save_dir, "splits.json"), 'w') as f:
        json.dump(splits, f)
    
    print("Preprocessing metadata saved to data/flickr8k/")

if __name__ == "__main__":
    main()
