import re
import json
from tokenizers.implementations import ByteLevelBPETokenizer


def fix_apostrophe_space(input_string):
    pattern = re.compile(r"(?<=\w)'\s(?=[tslvrm])")
    output_string = re.sub(pattern, "'", input_string)

    return output_string


def load_data():
    text = ""

    with open("/kaggle/input/machine-translation-data-set/enlish_data.txt", 'r', encoding='utf-8') as f:
        a_book = f.read()
    text += a_book

    text = fix_apostrophe_space(text)

    return text


def load_bpe():
    with open("bpe-tokenizer/bpe-using-library/bpe/vocab.json") as f:
        vocab = json.load(f)
#     with open("/kaggle/input/bpe-using-library/merges.txt") as f:
#         merges = f.read()

    tokenizer = ByteLevelBPETokenizer(
        "bpe-tokenizer/bpe-using-library/bpe/vocab.json",
        "bpe-tokenizer/bpe-using-library/bpe/merges.txt",
    )

    return vocab, tokenizer


def tokenize(text):
    start = 0
    end = int(100e6)
    offset = int(100e6)
    total_length = len(text)
    bpe = []

    vocab, tokenizer = load_bpe()

    while start != end:
        batch = text[start:end]
        pattern = re.compile(r'(.*)[\.?!\n]', re.DOTALL)
        match = pattern.search(batch)

        if match:
            extracted_text = match.group()
        else:
            extracted_text = ''

        end = end - (len(batch) - len(extracted_text))

        splits = tokenizer.encode(extracted_text).tokens
        bpe.append(splits)

        start = end
        end = end+offset if end+offset <= total_length else total_length

    bpe = sum(bpe, [])
    encode = lambda x: [vocab[_] for _ in x]
    input_ids = torch.tensor(encode(bpe))

    return input_ids, len(vocab)


def main():
    text = load_data()
    input_ids, vocab_size = tokenize(text)

    tokenized_data = {
        'input_ids': input_ids,
        'vocab_size': vocab_size
    }

    # torch.save(tokenized_data, 'bpe/tokenized_data.pt')
    torch.save(tokenized_data, '<enter_path>')


if __name__ == "__main__":
    main()
