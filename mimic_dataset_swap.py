import os
import torch
import json
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate as pytorch_default_collate
import random
from einops import rearrange

from paths import IMAGES_MIMIC_PATH, DICT_CSV_MIMIC_PATH
from mydatasets.mydatasets_utils import ifcc_clean_report, vilmedic_collate

class mimic_Dataset(Dataset):

    def __init__(self, 
                 transform, 
                 tokenizer,
                 processor,
                 partition = "train",
                 text_preprocessing="ifcc_clean_report",
                 multi_image=2):

        self.transform = transform
        self.tokenizer = tokenizer
        self.processor = processor
        self.partition = partition
        self.text_preprocessing = text_preprocessing if text_preprocessing is None else eval(text_preprocessing)
        self.multi_image = multi_image
        self.random_padding = self.partition == "train"

        # Load CSV partition
        self.csv_path = DICT_CSV_MIMIC_PATH[self.partition]
        self.dataset_df = pd.read_csv(self.csv_path)
        
        # 'swapped_answer' 열이 없는 경우를 대비해 빈 문자열로 채웁니다 (주로 test셋에 없을 수 있음)
        if 'swapped_answer' not in self.dataset_df.columns:
            self.dataset_df['swapped_answer'] = ""

        # Remove empty question or answer from self.dataset_df
        self.remove_empty_text()

        # Set images path
        self.img_root_dir = pathlib.Path(IMAGES_MIMIC_PATH) if IMAGES_MIMIC_PATH is not None else pathlib.Path.cwd()

        self.bos_token = self.tokenizer("[BOS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.eos_token = self.tokenizer("[EOS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.pad_token = self.tokenizer("[PAD]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.sep_token = self.tokenizer("[SEP]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.cls_token = self.tokenizer("[CLS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        print("BOS token: ", self.bos_token.item())
        print("EOS token: ", self.eos_token.item())
        print("PAD token: ", self.pad_token.item())
        print("SEP token: ", self.sep_token.item())
        print("CLS token: ", self.cls_token.item())

    def __len__(self):
        return len(self.dataset_df)
    
    def clean_bad_ids_rg(self):
        print("Initial number of rows: ", self.dataset_df.shape[0])
        self.l_no_ids_rg = []
        with open(self.path_no_ids_rg, 'r') as file:
            for line in file:
                # Process each line as needed
                self.l_no_ids_rg.append(int(line.strip()))
                
        self.dataset_df.drop(self.l_no_ids_rg, inplace=True)
        print("Number of rows after deleting bad IDs: ", self.dataset_df.shape[0])


    def __getitem__(self, idx):
        img_list_from_idx = []
        num_images = len(self.dataset_df.iloc[idx].images.split(","))

        for i in range(num_images):
            img_name = self.img_root_dir / self.dataset_df.iloc[idx].images.split(",")[i]
            image = Image.open(img_name).convert('RGB')
            
            if isinstance(self.transform, transforms.Compose):
                image = self.transform(image)
            elif isinstance(self.transform, A.core.composition.Compose):
                image = self.transform(image=np.asarray(image))['image']
            else:
                raise ValueError("Unknown transformation type.")
            
            image = np.array(image)
            image = self.processor(image, 
                                   random_padding=self.random_padding, 
                                   return_tensors="pt",
                                   size=384).pixel_values
            image = image.squeeze()
            img_list_from_idx.append(image)

        # Question
        question = self.dataset_df.iloc[idx].question
        raw_question = self.text_preprocessing(question)
        
        # Original Answer
        answer = self.dataset_df.iloc[idx].answer
        raw_answer = self.text_preprocessing(answer)

        # Swapped Answer
        swapped_answer_text = self.dataset_df.iloc[idx].swapped_answer
        if pd.isna(swapped_answer_text):
            swapped_answer_text = ""
        raw_swapped_answer = self.text_preprocessing(swapped_answer_text)

        # Tokenize Question
        question = self.tokenizer(raw_question,
                                  padding=False,
                                  truncation=True,
                                  max_length=64,
                                  return_tensors="pt",
                                  add_special_tokens=False)["input_ids"][0]
        
        question = torch.nn.functional.pad(question, (0, 64 - len(question) - 1), value=self.pad_token.item())
        question = torch.cat([self.cls_token, question, self.bos_token])
        question_mask = torch.ones_like(question)
        question_mask[question == self.pad_token] = 0

        # Tokenize Original Answer
        answer_tokens = self.tokenizer(
            raw_answer,
            padding=False,
            truncation=True,
            max_length=64,
            return_tensors="pt",
            add_special_tokens=False)["input_ids"][0]
        
        answer_tokens = torch.nn.functional.pad(answer_tokens, (0, 64 - len(answer_tokens) - 1), value=self.pad_token.item())
        answer_tokens = torch.cat([answer_tokens, self.eos_token])
        answer_mask = torch.ones_like(answer_tokens)
        answer_mask[answer_tokens == self.pad_token] = 0
        
        # Tokenize Swapped Answer
        swapped_answer_tokens = self.tokenizer(
            raw_swapped_answer,
            padding=False,
            truncation=True,
            max_length=64,
            return_tensors="pt",
            add_special_tokens=False)["input_ids"][0]

        swapped_answer_tokens = torch.nn.functional.pad(swapped_answer_tokens, (0, 64 - len(swapped_answer_tokens) - 1), value=self.pad_token.item())
        swapped_answer_tokens = torch.cat([swapped_answer_tokens, self.eos_token])
        swapped_answer_mask = torch.ones_like(swapped_answer_tokens)
        swapped_answer_mask[swapped_answer_tokens == self.pad_token] = 0

        question_ignore = torch.tensor([-100] * len(question))

        # Create labels for both answers
        labels = torch.cat([question_ignore, answer_tokens])
        swapped_labels = torch.cat([question_ignore, swapped_answer_tokens])

        im_and_immask = vilmedic_collate([img_list_from_idx], self.multi_image)
        images = im_and_immask["images"]
        images_mask = im_and_immask["images_mask"]
                  
        return {
            'idx': idx, 
            'images': images,
            'images_mask': images_mask,
            'question': question,
            'question_mask': question_mask,
            'answer': answer_tokens, # Renamed from answer to answer_tokens for clarity
            'answer_mask': answer_mask,
            'raw_question': raw_question,
            'raw_answer': raw_answer,
            'labels': labels,
            'swapped_answer': swapped_answer_tokens, # Added
            'swapped_answer_mask': swapped_answer_mask, # Added
            'raw_swapped_answer': raw_swapped_answer, # Added
            'swapped_labels': swapped_labels, # Added
            'main_image_path': self.dataset_df.iloc[idx].images.split(",")[0],
            'ref_image_path': self.dataset_df.iloc[idx].images.split(",")[1]
        }
    
    def remove_empty_text(self):
        self.dataset_df.dropna(subset=['question', 'answer'], inplace=True)
        print("Len before removing empty text", len(self.dataset_df))

        self.dataset_df = self.dataset_df[
            (self.dataset_df['question'].str.strip() != '') & 
            (self.dataset_df['answer'].str.strip() != '')
        ]
        print("Len after removing empty text", len(self.dataset_df))

    def get_collate_fn(self):
        def collate_fn(batch):
            images = pytorch_default_collate([s['images'] for s in batch])
            images_mask = pytorch_default_collate([s['images_mask'] for s in batch])
            idx = pytorch_default_collate([s['idx'] for s in batch])
            question = pytorch_default_collate([s['question'] for s in batch])
            question_mask = pytorch_default_collate([s['question_mask'] for s in batch])
            answer = pytorch_default_collate([s['answer'] for s in batch])
            answer_mask = pytorch_default_collate([s['answer_mask'] for s in batch])
            raw_question = [s['raw_question'] for s in batch]
            raw_answer = [s['raw_answer'] for s in batch]
            labels = pytorch_default_collate([s['labels'] for s in batch])
            main_image_path = [s['main_image_path'] for s in batch]
            ref_image_path = [s['ref_image_path'] for s in batch]
            
            # --- Start: Added for swapped data ---
            swapped_answer = pytorch_default_collate([s['swapped_answer'] for s in batch])
            swapped_answer_mask = pytorch_default_collate([s['swapped_answer_mask'] for s in batch])
            raw_swapped_answer = [s['raw_swapped_answer'] for s in batch]
            swapped_labels = pytorch_default_collate([s['swapped_labels'] for s in batch])
            # --- End: Added for swapped data ---

            collated = {
                'idx': idx,
                'images': images,
                'images_mask': images_mask,
                'questions_ids': question,
                'questions_mask': question_mask,
                'answers_ids': answer,
                'answers_mask': answer_mask,
                'questions': raw_question,
                'answers': raw_answer,
                'labels': labels,
                # --- Start: Added for swapped data ---
                'main_image_paths': main_image_path,
                'ref_image_paths': ref_image_path,
                'swapped_answers_ids': swapped_answer,
                'swapped_answers_mask': swapped_answer_mask,
                'swapped_answers': raw_swapped_answer,
                'swapped_labels': swapped_labels
                # --- End: Added for swapped data ---
            }
            return collated
        return collate_fn