from transformers import AutoTokenizer


def build_tokenize_fn(model_name_or_path, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        all_labels = []
        for i, label in enumerate(batch["labels"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
                prev_word_id = word_id
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    return tokenizer, tokenize_and_align
