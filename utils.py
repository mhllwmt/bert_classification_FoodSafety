class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convent_feature(train_words, tokenizer, max_seq):
    features = []
    for segment in train_words:
        words_token = tokenizer.tokenize(segment)
        if len(words_token) > max_seq:
            words_token = words_token[:max_seq]
            words_token = ['[CLS]'] + words_token + ['[SEP]']
        else:
            words_token = ['[CLS]'] + words_token + ['[SEP]']
            words_token += ['[PAD]'] * (max_seq + 2 - len(words_token))
        segment_ids = [0] * len(words_token)
        input_mask = [0 if token == '[PAD]' else 1 for token in words_token]
        input_ids = tokenizer.convert_tokens_to_ids(words_token)
        assert len(words_token) == max_seq + 2
        features.append(InputFeatures(input_ids, input_mask,segment_ids))
    return features

