from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._utils import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

from torchtune.models.aura import AuraDecoderTokenizer, AuraEncoderTokenizer

import os
import torch

# For our dataset, we will have a jsonl file with the following format:
# {
#     "transcript_text": "Hi how are you?",
#     "audio_vq_code_path": "abs_path/to/audio_code.pt",
#     "audio_description_text": "text description of the audio"
# }
class AuraDataset(Dataset):
    def __init__(self, decoder_tokenizer: AuraDecoderTokenizer, encoder_tokenizer: AuraEncoderTokenizer, data_path: str, split: str="train"):
        jsonl_path = os.path.join(data_path, f'{split}.jsonl')
        self.data = load_dataset('json', data_files=jsonl_path)['train']
        
        self.length = len(self.data)

        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_tokenizer = encoder_tokenizer

        self.pad_token_id = decoder_tokenizer.pad_token_id   
        self.encoder_tokenizer = encoder_tokenizer
   
        self.speech_generation_start_id = decoder_tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.speech_generation_end_id = decoder_tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        self.text_generation_start_id = decoder_tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
        self.text_generation_end_id = decoder_tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
        self.text_understanding_start_id = decoder_tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')
        self.text_understanding_end_id = decoder_tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')
        self.speech_understanding_start_id = decoder_tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
        self.speech_understanding_end_id = decoder_tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
 
        self.max_length = 2048
        self.ignore_index = -100  
        self.encoder_max_length = 1024
    def __len__(self):
        return self.length

    def pad_sequence(self, sequence, max_length, value=0):
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = torch.full((max_length - len(sequence),), value, dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=0)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # Load audio VQ codes and ensure they have shape [T]
        input_ids = torch.load(item['audio_vq_code_path'])
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze()
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0)
        assert input_ids.dim() == 1, f"Expected input_ids to have shape [T], got {input_ids.shape}"

        transcript_text = item['transcript_text']
        audio_description_text = item['audio_description_text']
        # Validate that required fields are present
        if input_ids is None or len(input_ids) == 0:
            raise ValueError(f"Missing or empty audio VQ codes at index {idx}")

        if not transcript_text or transcript_text.strip() == "":
            raise ValueError(f"Missing or empty transcript_text at index {idx}")
            
        if not audio_description_text or audio_description_text.strip() == "":
            raise ValueError(f"Missing or empty audio_description_text at index {idx}")
            
        audio_codes_string = self.decoder_tokenizer.convert_xcodec_tokens_to_string(input_ids)
        chat = [
            {"role": "user", "content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{transcript_text}<|TEXT_UNDERSTANDING_END|>"},
            {"role": "assistant", "content": f"<|SPEECH_GENERATION_START|>{audio_codes_string}<|SPEECH_GENERATION_END|>"}
        ]
        ids = self.decoder_tokenizer.apply_chat_template(chat, tokenize=True)

        decoder_input_ids = torch.tensor(ids, dtype=torch.long)
        decoder_labels = torch.full_like(decoder_input_ids, self.ignore_index)

        try:
            speech_gen_idx_in_input = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0].item()
            decoder_labels[speech_gen_idx_in_input:] = input_ids[speech_gen_idx_in_input:]
        except Exception as e:
            print(f"maybe Error in speech_gen_idx_in_input: {e}")
            # speech_gen_idx_in_input = len(input_ids) - 1
            decoder_labels = input_ids 

        decoder_attention_mask = (input_ids != self.pad_token_id).long()
        decoder_labels[input_ids == self.pad_token_id] = self.ignore_index


        encoder_inputs = self.encoder_tokenizer(
            audio_description_text,
            max_length=self.encoder_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Remove the batch dimension added by the tokenizer
        encoder_input_ids = encoder_inputs['input_ids'].squeeze(0)
        encoder_attention_mask = encoder_inputs['attention_mask'].squeeze(0) 
        
        decoder_input_ids = self.pad_sequence(decoder_input_ids, self.max_length, value=self.pad_token_id)
        decoder_attention_mask = self.pad_sequence(decoder_attention_mask, self.max_length, value=0)
        decoder_labels = self.pad_sequence(decoder_labels, self.max_length, value=self.ignore_index)

        return {
            'decoder_input_ids': list(decoder_input_ids),
            'decoder_labels': list(decoder_labels),
            'decoder_attention_mask': list(decoder_attention_mask),
            'encoder_input_ids': list(encoder_input_ids),
            'encoder_attention_mask': list(encoder_attention_mask)
        }
 