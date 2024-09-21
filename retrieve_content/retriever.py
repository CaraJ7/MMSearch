from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

from retrieve_content.tokenization import tokenizers
from dataclasses import dataclass
from typing import Optional
import torch

JOIN_WITH_WHITESPACE: str = "join_with_whitespace"

@dataclass
class Config:
    num_threads: int = 5
    docs_limit: Optional[int] = None
    docs_offset: Optional[int] = None
    chunk_length: int = 256
    slidew: bool = False
    sentb: bool = False
    title: bool = False
    paragraphs: bool = False
    detokenization_strategy: str = JOIN_WITH_WHITESPACE
    TopK: int = 8


class Content_Retriever:
    def __init__(self):
        # define tokenizer
        self.tokenizer = tokenizers.WordTokenizer()
        self.tokenizer_offsets = tokenizers.WordTokenizer(do_char_offsets=True)

        self.config = Config()

        self.tokenizer_offsets.passage_len = self.config.chunk_length
        self.tokenizer_offsets.do_sliding_window_passages = self.config.slidew
        self.tokenizer_offsets.respect_sent_boundaries = self.config.sentb
        # define retrieval model
        self.model = BGEM3FlagModel(
            'BAAI/bge-m3',  
            use_fp16=True
        ) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def split_doc_into_passages(self, doc, detokenization_strategy: str = JOIN_WITH_WHITESPACE):
        text = doc
        passages = []
        if detokenization_strategy == JOIN_WITH_WHITESPACE:
            passages_tokens = self.tokenizer.tokenize_passages(text)
            for passage_idx, passage_tokens in enumerate(passages_tokens):
                if self.tokenizer.respect_sent_boundaries:
                    # passages_tokens: List[List[str]]
                    tokens = []
                    for psg in passage_tokens:
                        tokens.extend(psg)
                    passage_tokens = tokens
                if len(passage_tokens) == 0:
                    continue
                # passages_tokens: List[str]
                passage_text = " ".join(passage_tokens)
                passages.append(passage_text)
        
        return passages

    def try_split_passages(self, doc):
        try:
            return self.split_doc_into_passages(doc, self.config.detokenization_strategy)
        except ValueError as e:
            print('ValueError:', e)
            return None
    
    def get_retrieved_content(self, requery, content):
        docs = [content]
        all_chucks = self.try_split_passages(content)
        # encode
        output_1 = self.model.encode([requery], return_dense=True, return_sparse=True, return_colbert_vecs=True, batch_size=12, max_length=self.config.chunk_length)
        output_2 = self.model.encode(all_chucks, return_dense=True, return_sparse=True, return_colbert_vecs=True, batch_size=12, max_length=self.config.chunk_length)
        scores = []
        for i in range(len(output_2['colbert_vecs'])):
            scores.append(
                self.model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][i]).item()
            )

        sorted_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        sorted_values, original_indices = zip(*sorted_pairs)
        return '\n'.join([all_chucks[idx] for idx in sorted_values[:self.config.TopK]])

