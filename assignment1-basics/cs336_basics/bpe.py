import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
from multiprocessing import Pool


class BPE:
    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_precessor: int = 4,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
    ):
        self.vocab_size = vocab_size
        self.input_path = input_path
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = {i: bytes([i]) for i in range(256)}
            for i, token in enumerate(special_tokens):
                self.vocab[256 + i] = token.encode("utf-8")

        if merges:
            self.merges = merges
        else:
            self.merges = []

        self.special_tokens = special_tokens
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.regex = re.compile(self.pat)
        self.special_pat = "|".join(map(re.escape, self.special_tokens))
        self.special_regex = re.compile(self.special_pat)
        self.num_precessor = num_precessor

    # 1. Pre-tokenization
    # 2. Apply BPE merges
    def encode(self, input_path: str | os.PathLike) -> list[int]:
        res = []
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_precessor, b"<|endoftext|>")

        return res

    # 1. Pre-tokenization
    # 2. Compute BPE merges
    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.num_merges = self.vocab_size - len(self.special_tokens) - 256
        assert self.num_merges > 0
        assert self.input_path

        word_freqs = self._pre_tokenization()
        merges = self._merge(word_freqs)
        return self.vocab, merges

    def _merge(self, word_freqs: dict[tuple[bytes], int]) -> list[tuple[bytes, bytes]]:
        merges = []

        for _ in range(self.num_merges):
            pair_freqs = defaultdict(int)
            for tokens, freq in word_freqs.items():
                for pair in zip(tokens, tokens[1:]):
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
            best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))

            # Create new merged token
            merged_token = best_pair[0] + best_pair[1]

            # Add to vocabulary and merges
            self.vocab[len(self.vocab)] = merged_token
            merges.append(best_pair)

            # Update all word token sequences
            new_word_freqs = defaultdict(int)
            for tokens, freq in word_freqs.items():
                new_tokens = []
                i = 0
                n = len(tokens)
                while i < n:
                    if i < n - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_word_freqs[tuple(new_tokens)] += freq
            word_freqs = new_word_freqs

        self.merges = merges
        return merges

    def _process_chunk(self, input_path, start, end):
        word_freqs = defaultdict(int)

        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="replace")

        parts = self.special_regex.split(chunk)
        for part in parts:
            for match in self.regex.finditer(part):
                word = match.group()
                word_bytes = word.encode("utf-8")
                tokens = tuple(self.vocab[b] for b in word_bytes)
                word_freqs[tokens] += 1

        return word_freqs

    def _pre_tokenization(self) -> dict[tuple[bytes], int]:
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_precessor, b"<|endoftext|>")

        tasks = [
            (
                self.input_path,
                start,
                end,
            )
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(self.num_precessor) as pool:
            results = pool.starmap(self._process_chunk, tasks)

        word_freqs = defaultdict(int)
        for partial_dict in results:
            for k, v in partial_dict.items():
                word_freqs[k] += v

        return word_freqs

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


if __name__ == "__main__":
    # bpe = BPE(
    #     input_path="tests/fixtures/corpus.en",
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )
    # v, m = bpe.train()

    # pair_freqs = defaultdict(int)
    # pair_freqs[("ba", "a")] = 1
    # pair_freqs[("b", "zz")] = 1
    # best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
    # print(best_pair)

    # c = "こ"
    # encoded = c.encode("utf-8")
    # print([tuple(bytes([i]) for i in encoded)])

    text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    # print(regex.findall(text))

    special_tokens = ["<|endoftext|>"]
    # special_regex = re.compile("|".join(map(re.escape, special_tokens)))
    # print(special_regex.split(text))

    special_pat = f"(?P<special>{'|'.join(map(re.escape, special_tokens))})"
    pat = r"(?P<normal>'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"
    combined_pat = f"{special_pat}|{pat}"
    combined_regex = re.compile(combined_pat)
    print(combined_pat)

    text1 = "Hello world <|endoftext|>"

    for match in combined_regex.finditer(text1):
        if match.lastgroup == "special":
            print(f"SPECIAL : {match.group('special')}")
        elif match.lastgroup == "normal":
            print(f"NORMAL  : {match.group('normal')}")

    pass
