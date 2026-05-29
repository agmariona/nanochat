import pytest

from nanochat.tokenizer import ByteTokenizer, SPECIAL_TOKENS, get_token_bytes

@pytest.mark.parametrize("text",
    ["hello", "café", "🙂", "", "\n", "\x00", "a\nb"])
def test_byte_tokenizer_roundtrip(text):
    tok = ByteTokenizer()
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert all(0 <= i < tok.get_vocab_size() for i in ids)

def test_byte_tokenizer_arbitrary():
    tok = ByteTokenizer()
    tok.decode([255])
    tok.decode([0xC3])

def test_byte_tokenizer_special_tokens():
    tok = ByteTokenizer()
    ids = [
        tok.get_bos_token_id(),
        *tok.encode("hi"),
        tok.encode_special("<|user_start|>")
    ]
    assert tok.decode(ids) == "hi"
    assert tok.id_to_token(tok.get_bos_token_id()) == "<|bos|>"

def test_byte_tokenizer_vocab_and_special_ids():
    tok = ByteTokenizer()
    assert tok.get_vocab_size() == 256 + len(SPECIAL_TOKENS)

    ids = [tok.encode_special(s) for s in SPECIAL_TOKENS]
    assert len(ids) == len(set(ids))
    assert min(ids) >= 256
    assert max(ids) < tok.get_vocab_size()
    assert tok.get_bos_token_id() == tok.encode_special("<|bos|>")

def test_byte_tokenizer_batch_encode_with_prepend():
    tok = ByteTokenizer()
    bos = tok.get_bos_token_id()
    rows = tok.encode(["hi", "bye"], prepend=bos, num_threads=4)

    assert isinstance(rows, list)
    assert all(isinstance(row, list) for row in rows)
    assert [row[0] for row in rows] == [bos, bos]
    assert tok.decode(rows[0]) == "hi"
    assert tok.decode(rows[1]) == "bye"

def test_byte_tokenizer_prepend_append_string_specials():
    tok = ByteTokenizer()
    ids = tok.encode("hi", prepend="<|bos|>", append="<|assistant_end|>")

    assert ids[0] == tok.encode_special("<|bos|>")
    assert ids[-1] == tok.encode_special("<|assistant_end|>")
    assert tok.decode(ids) == "hi"

def test_byte_tokenizer_call_alias():
    tok = ByteTokenizer()
    assert tok("hello") == tok.encode("hello")

def test_byte_tokenizer_utf8_uses_bytes():
    tok = ByteTokenizer()
    ids = tok.encode("é")
    assert ids == list("é".encode("utf-8"))
    assert len(ids) == 2

def test_byte_tokenizer_unknown_invalid():
    tok = ByteTokenizer()
    with pytest.raises(KeyError):
        tok.encode_special("<|not_real|>")
    with pytest.raises(ValueError):
        tok.decode([tok.get_vocab_size()])
    with pytest.raises(ValueError):
        tok.decode([-1])

def test_byte_tokenizer_token_bytes():
    tok = ByteTokenizer()
    token_bytes = get_token_bytes(tok)
    assert token_bytes.shape == (tok.get_vocab_size(),)
    assert token_bytes[:256].eq(1).all()
    assert token_bytes[256:].eq(0).all()
