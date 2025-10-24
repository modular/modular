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

from max.pipelines.lib import TextAndVisionTokenizer

# https://docs.modular.com/max/api/python/pipelines/tokenizer/#max.pipelines.lib.tokenizer.TextAndVisionTokenizer
class Gemma3VLTokenizer(TextAndVisionTokenizer):
    def __init__(
        self,
    ) -> None:
    # this class is only required if we need custom behavior