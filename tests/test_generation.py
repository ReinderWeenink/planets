import pytest
import random
import string
from hypothesis import given, strategies as st, settings, Verbosity
from unittest.mock import patch, MagicMock

# Import the function we want to test
import backend.app
from backend.app import new_words

# -----------------------------------------------------------------------------
# 1. Define Strategies
# -----------------------------------------------------------------------------
num_words_strategy = st.integers(min_value=1, max_value=100)
temp_strategy = st.floats(min_value=0.01, max_value=10.0)

# -----------------------------------------------------------------------------
# 2. Mock Generator
# -----------------------------------------------------------------------------
def mock_sample_n_implementation(n, model, tokenizer, max_length, temperature):
    results = []
    for _ in range(n):
        current_len = random.randint(1, max_length)
        random_word = "".join(random.choices(string.ascii_lowercase, k=current_len))
        results.append(random_word)
    return results

# -----------------------------------------------------------------------------
# 3. Property-Based Test
# -----------------------------------------------------------------------------
@given(n=num_words_strategy, temperature=temp_strategy)
@settings(max_examples=50, verbosity=Verbosity.normal)
def test_generation_properties(n, temperature):
    """
    Property-based test using Hypothesis.
    """
    # Create the mock state we want to inject
    fake_state = {
        "model": MagicMock(),
        "tokenizer": MagicMock()
    }

    # Use patch.dict to forcibly update the global model_state dictionary
    # just for the duration of this specific test case execution.
    with patch.dict(backend.app.model_state, fake_state):
        # We also patch 'sample_n' to use our simulator
        with patch("backend.app.sample_n", side_effect=mock_sample_n_implementation):
            
            words = new_words(n, temperature)
            
            # --- Property 1: Type Safety ---
            assert isinstance(words, list), "Output must be a list"
            assert all(isinstance(w, str) for w in words), "All generated items must be strings"
            
            # --- Property 2: Correct Count ---
            assert len(words) == n, f"Requested {n} words, but got {len(words)}"
            
            # --- Property 3: Constraints (Max Length) ---
            HARDCODED_MAX_LENGTH = 20
            for word in words:
                assert len(word) <= HARDCODED_MAX_LENGTH, \
                    f"Word '{word}' is {len(word)} chars long, exceeding limit."