# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Dict, List, Union, Optional, Tuple


import torch

class AuraDecoderTokenizer():
    def __init__(
        self,
        path: str = "HKUSTAudio/Llasa-1B",
        max_seq_len: int = 2048,
        truncation_type: str = "right",
    ):
        from transformers import AutoTokenizer
        
        # Initialize the HuggingFace tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            model_max_length=max_seq_len if max_seq_len is not None else 2048,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token  # For LLaMa
        original_vocab_size = len(self.tokenizer)
        print(f"Original tokenizer vocabulary size: {original_vocab_size}")

        # Store special token IDs
        self.max_seq_len = max_seq_len
        
        # Set up token IDs
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        
        
        self.truncation_type = truncation_type

    def apply_chat_template(self, chat: List[Dict[str, str]], tokenize: bool = True) -> List[int]:
        """
        Apply the chat template to a list of messages and optionally tokenize the result.
        
        Args:
            chat (List[Dict[str, str]]): A list of message dictionaries with 'role' and 'content' keys.
            tokenize (bool): Whether to tokenize the result. Default is True.
            
        Returns:
            List[int] or str: The tokenized result as a list of token IDs if tokenize=True,
                             otherwise the formatted chat string.
        """
        # Use the underlying tokenizer's apply_chat_template method
        return self.tokenizer.apply_chat_template(
            chat, 
            tokenize=tokenize
        )
    
    def convert_tokens_to_ids(self, tokens):
        """
        Convert a token or a list of tokens to their corresponding token IDs.
        
        Args:
            tokens (str or List[str]): A token or a list of tokens to convert to IDs.
            
        Returns:
            int or List[int]: The token ID or list of token IDs.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_xcodec_tokens_to_string(self, tokens):
        """
        Convert a tensor of xcodec tokens to a string representation.
        
        Args:
            tokens (torch.Tensor): A tensor of shape [T] containing token IDs.
            
        Returns:
            str: A string representation of the tokens in the format '<|s_token_id|>'.
        """
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        
        result = ""
        for token in tokens:
            result += f"<|s_{token.item()}|>"
        
        return result

class AuraEncoderTokenizer:
    """
    Tokenizer for the Aura encoder model based on DistilBERT.
    
    This tokenizer handles the text encoding part of the Aura model,
    using DistilBERT's tokenizer for efficient text understanding.
    """
    
    def __init__(
        self,
        path: str = "distilbert-base-uncased",
        max_seq_len: Optional[int] = None,
        truncation_type: str = "right",
    ):
        """
        Initialize the AuraEncoderTokenizer.
        
        Args:
            path (str): Path or name of the pretrained tokenizer to use.
                Default is "distilbert-base-uncased".
            max_seq_len (Optional[int]): Maximum sequence length for the tokenizer.
                If None, defaults to the model's max length or 512.
            truncation_type (str): Direction of truncation ("left" or "right").
                Default is "right".
        """
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            model_max_length=max_seq_len if max_seq_len is not None else 512,
            padding_side="right",
        )
        
        original_vocab_size = len(self.tokenizer)
        print(f"Encoder tokenizer vocabulary size: {original_vocab_size}")
        
        # Store special token IDs and configuration
        self.max_seq_len = max_seq_len
        self.truncation_type = truncation_type
        
        # Set up token IDs
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.mask_id = self.tokenizer.mask_token_id
    
    def encode(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Encode text into input_ids, attention_mask, and token_type_ids.
        
        Args:
            text (Union[str, List[str]]): Text or list of texts to encode.
            **kwargs: Additional arguments to pass to the tokenizer.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the encoded inputs.
        """
        encoding = self.tokenizer(
            text,
            padding="max_length" if self.max_seq_len else True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
            **kwargs
        )
        return encoding
    
    def decode(self, token_ids: Union[torch.Tensor, List[int]], **kwargs) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (Union[torch.Tensor, List[int]]): Token IDs to decode.
            **kwargs: Additional arguments to pass to the tokenizer.
            
        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def convert_tokens_to_ids(self, tokens):
        """
        Convert a token or a list of tokens to their corresponding token IDs.
        
        Args:
            tokens (str or List[str]): A token or a list of tokens to convert to IDs.
            
        Returns:
            int or List[int]: The token ID or list of token IDs.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)