# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Custom tokenizer for BERT that ensures EOS token is available."""

from max.pipelines.lib import TextTokenizer


class BertTextTokenizer(TextTokenizer):
    """Custom TextTokenizer for BERT that ensures EOS token is properly configured."""

    def __init__(self, *args, **kwargs):
        # Initialize cached EOS first to avoid AttributeError during parent init
        self._cached_eos = 102  # Default to [SEP] token ID for BERT
        super().__init__(*args, **kwargs)
        self._configure_eos_token()
        # Update the cached EOS token ID after configuration
        self._cached_eos = self._get_eos_token_id()

    def _configure_eos_token(self):
        """Configure EOS token for BERT models that don't have one."""
        # Try different ways to access the underlying HF tokenizer
        hf_tokenizer = None
        if hasattr(self, "tokenizer"):
            hf_tokenizer = self.tokenizer
        elif hasattr(self, "_tokenizer"):
            hf_tokenizer = self._tokenizer

        # If we can access the HF tokenizer and it lacks EOS token
        if hf_tokenizer is not None:
            if getattr(hf_tokenizer, "eos_token", None) is None:
                # Use SEP token as EOS token for BERT
                sep_token = getattr(hf_tokenizer, "sep_token", None)
                sep_token_id = getattr(hf_tokenizer, "sep_token_id", None)

                if sep_token is not None:
                    hf_tokenizer.eos_token = sep_token
                    if sep_token_id is not None:
                        hf_tokenizer.eos_token_id = sep_token_id

                    # Update special tokens map if it exists
                    if hasattr(hf_tokenizer, "special_tokens_map"):
                        hf_tokenizer.special_tokens_map["eos_token"] = sep_token

        # Also ensure the wrapper exposes the eos property correctly
        if hasattr(self, "_update_eos"):
            self._update_eos()

    def _get_eos_token_id(self):
        """Get the EOS token ID, falling back to SEP token ID for BERT."""
        # Try different ways to access the underlying HF tokenizer
        hf_tokenizer = getattr(self, "tokenizer", None) or getattr(
            self, "_tokenizer", None
        )

        if hf_tokenizer is not None:
            # Try EOS token first
            eos_id = getattr(hf_tokenizer, "eos_token_id", None)
            if eos_id is not None:
                return eos_id

            # Fall back to SEP token for BERT
            sep_id = getattr(hf_tokenizer, "sep_token_id", None)
            if sep_id is not None:
                return sep_id

        # Default fallback - use token ID 102 which is typically [SEP] for BERT
        return 102

    @property
    def eos(self):
        """Get EOS token ID, using SEP token if EOS is not available."""
        # Return cached value that is guaranteed to be non-None
        return self._cached_eos
