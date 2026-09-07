from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from sectrace.tokenization import train_tokenizer, CodeTokenizer

@pytest.fixture(scope="session", autouse=True)
def bounded_cpu_threads():
    torch.set_num_threads(2)

@pytest.fixture
def tokenizer(tmp_path):
    p = tmp_path / "byte.json"
    train_tokenizer([], str(p), diagnostic=True)
    return CodeTokenizer(p)
