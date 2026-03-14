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
        num_processor: int = 4,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
    ):
        self.vocab_size = vocab_size
        self.input_path = input_path
        if vocab:
            self.vocab = vocab
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        else:
            self.vocab = {i: bytes([i]) for i in range(256)}
            for i, token in enumerate(special_tokens):
                self.vocab[256 + i] = token.encode("utf-8")

        if merges:
            self.merges = merges
            self.merge_dict = {v: i for i, v in enumerate(merges)}
        else:
            self.merges = []

        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.regex = re.compile(self.pat)
        self.special_tokens = special_tokens
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.special_pat = "|".join(map(re.escape, self.special_tokens))
        self.special_regex = re.compile(self.special_pat) if self.special_pat else None
        self.num_processor = num_processor

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""

        chunk_size = max(1, len(ids) // self.num_processor)
        chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
        with Pool(self.num_processor) as pool:
            byte_chunks = pool.map(self._decode_chunk, chunks)

        combined_bytes = b"".join(byte_chunks)
        return combined_bytes.decode("utf-8", errors="replace")

    def _decode_chunk(self, chunk: list[int]) -> bytes:
        return b"".join(self.vocab[i] for i in chunk)

    from collections.abc import Iterable, Iterator

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            if self.special_tokens:
                parts = re.split(f"({self.special_pat})", text)
            else:
                parts = [text]

            for part in parts:
                if part in self.special_tokens:
                    yield self.reverse_vocab[part.encode("utf-8")]
                    continue

                for match in self.regex.finditer(part):
                    word = match.group()
                    word_bytes = word.encode("utf-8")

                    tokens = tuple(bytes([b]) for b in word_bytes)
                    yield from self._encode_to_tokens(tokens)

    # 1. Pre-tokenization
    # 2. Apply BPE merges
    def encode(self, text: str) -> list[int]:
        pre_tokens = []

        # keep special tokens using capture split
        if self.special_tokens:
            parts = re.split(f"({self.special_pat})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                pre_tokens.append(part.encode("utf-8"))
                continue

            for match in self.regex.finditer(part):
                word = match.group()
                pre_tokens.append(word.encode("utf-8"))

        chunk_size = max(1, len(pre_tokens) // self.num_processor)
        chunks = [pre_tokens[i : i + chunk_size] for i in range(0, len(pre_tokens), chunk_size)]

        with Pool(self.num_processor) as pool:
            results = pool.map(self._encode_chunk, chunks)

        return [t for chunk in results for t in chunk]

    def _encode_chunk(self, chunk: list[bytes]) -> list[int]:
        res = []

        for token_bytes in chunk:
            if token_bytes in self.reverse_vocab:
                res.append(self.reverse_vocab[token_bytes])
                continue

            tokens = tuple(bytes([b]) for b in token_bytes)
            res.extend(self._encode_to_tokens(tokens))

        return res

    def _encode_to_tokens(self, input: tuple[bytes, ...]) -> list[int]:
        while True:
            best_pair = None
            best_rank = float("inf")

            for pair in zip(input, input[1:]):
                cur = self.merge_dict.get(pair)
                if cur is not None and cur < best_rank:
                    best_rank = cur
                    best_pair = pair

            if best_pair is None:
                break

            new_input = []
            i = 0
            while i < len(input):
                if i < len(input) - 1 and (input[i], input[i + 1]) == best_pair:
                    new_input.append(input[i] + input[i + 1])
                    i += 2
                else:
                    new_input.append(input[i])
                    i += 1

            input = tuple(new_input)

        return [self.reverse_vocab[t] for t in input]

    def _encode_trunk(self, input_path: str | os.PathLike, start: int, end: int) -> list[int]:
        res = []

        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="replace")

        parts = self.special_regex.split(chunk) if self.special_regex else [chunk]
        for part in parts:
            if part in self.special_tokens:
                res.append(self.reverse_vocab[part.encode("utf-8")])
                continue

            for match in self.regex.finditer(part):
                word = match.group()
                word_bytes = word.encode("utf-8")
                tokens = self._encode_to_tokens(tuple(self.vocab[b] for b in word_bytes))
                res.extend(tokens)
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

            best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))

            # Create new merged token
            merged_token = best_pair[0] + best_pair[1]

            # Add to vocabulary and merges
            idx = len(self.vocab)
            self.vocab[idx] = merged_token
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
        self.merge_dict = {v: i for i, v in enumerate(merges)}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        return merges

    def _process_chunk(self, input_path: str | os.PathLike, start: int, end: int):
        word_freqs = defaultdict(int)

        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="replace")

        parts = self.special_regex.split(chunk) if self.special_regex else [chunk]
        for part in parts:
            for match in self.regex.finditer(part):
                word = match.group()
                word_bytes = word.encode("utf-8")
                tokens = tuple(self.vocab[b] for b in word_bytes)
                word_freqs[tokens] += 1

        return word_freqs

    def _pre_tokenization(self) -> dict[tuple[bytes], int]:
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_processor, b"<|endoftext|>")

        tasks = [
            (
                self.input_path,
                start,
                end,
            )
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(self.num_processor) as pool:
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
    # print([tuple(bytes([i]) for i in enoded)])

    # word = "hello"
    # word_bytes = word.encode("utf-8")
    # tokens = tuple(bytes([b]) for b in word_bytes)
    # print(tokens[0] + tokens[1])
    # print(type(tokens[0] + tokens[1]))

    text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    # print(regex.findall(text))

    special_tokens = ["<|endoftext|>"]
    special_pat = "|".join(map(re.escape, special_tokens))
    special_regex = re.compile(special_pat)
    print(special_regex.split(text))

    parts = re.split(f"({special_pat})", text)
    print(parts)

    pass
