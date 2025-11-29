import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from backend.app import load_model

class MockTokenizer:
    def encode(self, text):
        ids = [ord(c) for c in text]
        return MagicMock(ids=ids)
    
    def decode(self, ids):
        return "".join(chr(i) for i in ids)
    
    @classmethod
    def from_file(cls, path):
        return cls()

class TestModelLoading:
    """Tests for the load_model function including error handling."""

    @pytest.fixture
    def mock_artefacts_path(self):
        with patch("backend.app.ARTEFACTS", Path("/fake/path")) as p:
            yield p

    # Patching Path.open specifically because app.py uses config_path.open()
    @patch("pathlib.Path.open", new_callable=mock_open, read_data='{"model": {"some": "config"}}')
    @patch("backend.app.models.SlangRNN")
    @patch("backend.app.torch.load")
    # Patch the class itself, not just the method, to be safer
    @patch("backend.app.Tokenizer") 
    @patch("pathlib.Path.exists")
    def test_load_model_success(self, mock_exists, mock_tokenizer_cls, mock_torch, mock_rnn, mock_file, mock_artefacts_path):
        """Test that model and tokenizer load correctly when all files exist."""
        # Setup: All files exist
        mock_exists.return_value = True
        
        # Setup: Tokenizer.from_file returns our MockTokenizer instance
        mock_tokenizer_cls.from_file.return_value = MockTokenizer()
        
        model, tokenizer = load_model()
        
        assert model is not None
        assert isinstance(tokenizer, MockTokenizer)
        
        # Check if config was opened using Path.open
        assert mock_file.call_count >= 1

    @patch("backend.app.Tokenizer")
    @patch("pathlib.Path.exists")
    def test_missing_tokenizer_error(self, mock_exists, mock_tok_cls, mock_artefacts_path):
        """Test that missing tokenizer file raises FileNotFoundError."""
        # First check (tokenizer) returns False
        mock_exists.side_effect = [False]
        
        with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
            load_model()

    @patch("backend.app.Tokenizer")
    @patch("pathlib.Path.exists")
    def test_missing_config_error(self, mock_exists, mock_tok_cls, mock_artefacts_path):
        """Test that missing config file raises FileNotFoundError."""
        # Tokenizer exists (True), Config does not (False)
        mock_exists.side_effect = [True, False]
        # Ensure Tokenizer.from_file doesn't crash even if called
        mock_tok_cls.from_file.return_value = MockTokenizer()

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_model()

    # Patch Path.open for the invalid JSON test
    @patch("pathlib.Path.open", new_callable=mock_open, read_data='INVALID JSON')
    @patch("pathlib.Path.exists")
    @patch("backend.app.Tokenizer")
    def test_malformed_config_error(self, mock_tok_cls, mock_exists, mock_file, mock_artefacts_path):
        """Test that invalid JSON in config raises ValueError."""
        mock_exists.return_value = True
        mock_tok_cls.from_file.return_value = MockTokenizer()
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_model()

class TestTokenizerProperties:
    def test_tokenizer_reversibility(self):
        tokenizer = MockTokenizer()
        test_words = ["hello", "world", "slang", "test"]
        
        for word in test_words:
            encoded = tokenizer.encode(word)
            decoded = tokenizer.decode(encoded.ids)
            assert decoded == word