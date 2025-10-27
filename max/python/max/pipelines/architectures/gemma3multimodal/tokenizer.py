# image config
#  config for processing and mem estimation that are specific to the model
#  (may not be necessary?)

# visual processor
#    text tokenization and image token insertion
#    mimics the interface expected by TextAndVisionTokenizer
#  gemma3 may only need to implement TextAndVisionTokenizer directly
#  apply_chat_template wraps one from the Transformers package
#    seem to handle the actual API request
#  __call__ converts images to raw pixel vals (no processing) and arranges the
#  text input with the image tokens.  also sorts out image indices etc

# tokenizer
#  extends TextAndVisionTokenizer
#  overrides `__init__` to create a custom processor
#  we will make use of `new_context` etc from the parent class

# InternVL does stuff like selecting what the EOS, EOI, BOI tokens look like

# from max.pipelines.lib import TextAndVisionTokenizer
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

from .image_processing import Gemma3ImageProcessor

@dataclass(frozen=False, eq=True)
class AddedToken:
    """
    AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
    way it should behave.

    The `normalized` will default to `not special` if it is not specified, similarly to the definition in
    `tokenizers`.

    TODO: this was added manually because in HF transformers it's imported from tokenization_utils_base
    """

    def __init__(
        self, content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None
    ):
        self.content = content
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.special = special
        self.normalized = normalized if normalized is not None else not special

    def __getstate__(self):
        return self.__dict__

    def __str__(self):
        return self.content

# https://docs.modular.com/max/api/python/pipelines/tokenizer/#max.pipelines.lib.tokenizer.TextAndVisionTokenizer
# needs to implement SiglipTokenizer https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipTokenizer
# TODO not sure if it should be extending PreTrainedTokenizer/TextAndVisionTokenizer... internvl TextAndVisionTokenizer
# this may be relevant? https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/sentencepiece.i#L781
class Gemma3VisionTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: str,
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "</s>",
        additional_special_tokens: list[str] | None = None,
        sp_model_kwargs: dict | None = None,
        model_max_length: int = 64,
        do_lower_case: bool = False,
        **kwargs,
    ) -> None:
        # TODO tbh looks overkill
        pad_token = (
            str(AddedToken(pad_token, rstrip=True, lstrip=True, normalized=False, special=True))
            if isinstance(pad_token, str)
            else pad_token
        )
        unk_token = (
            str(AddedToken(unk_token, rstrip=True, lstrip=True, normalized=False, special=True))
            if isinstance(unk_token, str)
            else unk_token
        )
        eos_token = (
            str(AddedToken(eos_token, rstrip=True, lstrip=True, normalized=False, special=True))
            if isinstance(eos_token, str)
            else eos_token
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file
        
        # TODO no doubt this needs replacing with our own SiglipImageProcessor Gemma3SiglipProcessor
        # doesn't even seem to be doing anything
        # self.sp_model = self.get_spm_processor()
        self.vocab_file = vocab_file

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            model_max_length=model_max_length,
            do_lower_case=do_lower_case,
            **kwargs,
        )
    
    def _add_eos_if_not_present(self, token_ids: list[int]) -> list[int]:
        """Do not add eos again if user already added it.
        NOTE:  this wasn't mentioned on huggingface's description of Siglip, but the build inputs func needs it
        
        https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip/tokenization_siglip.py#L193
        """
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            return token_ids
        else:
            return token_ids + [self.eos_token_id]
    
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None
    ) -> list[int]:
        # TODO ai generated all this :D
        """Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Siglip sequence has the following format:
        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (`List[int]`): List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.

        https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip/tokenization_siglip.py#L228
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1
    
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False
    ) -> list[int]:
        # TODO ai generated all this :D
        """Retrieves sequence ids from a token list that has no special tokens added.

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs. 
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]:
            1 for a special token, 0 for a sequence token.

        https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip/tokenization_siglip.py#L164
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
    
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None
    ) -> list[int]:
        # TODO ai generated all this :D
        """Create a mask from the two sequences passed to be used in a sequence-pair 
        classification task. T5 does not make use of token type ids, therefore a
        list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).

        https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip/tokenization_siglip.py#L205
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str]:
        """https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip/tokenization_siglip.py#L362"""
        return ("",)