"""This code is a modified version provided at https://github.com/DhavalTaunk08/XWikiGen/tree/main"""
"""This module implements an extractive summarization approach based on masked language model (MLM) scoring. It extends Hugging Face's  `AutoModelForMaskedLM` 
with custom loss computation, then uses it to rank candidate reference sentences and select the top-k most relevant ones for each section of a document."""
import sys
import json
import regex
import argparse
from tqdm import tqdm
import torch
from polyglot.text import Text
from transformers import AutoTokenizer, AutoModelForMaskedLM
from indicnlp.tokenize.sentence_tokenize import sentence_split
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

class AutoModelForMaskedLMwithLoss(AutoModelForMaskedLM):
    """ Extension of Hugging Face's AutoModelForMaskedLM to compute token-level loss. This class overrides the forward method to add support for
    computing masked language modeling (MLM) loss at the token level, weighted by the attention mask."""
    def __init__(self, config):
        """Initialize the masked LM model with loss computation.
        Args:
            config (transformers.PretrainedConfig): Model configuration object."""
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
                """Forward pass with optional loss computation.
        Args:
            input_ids (torch.LongTensor): Token IDs of shape (batch_size, seq_length).
            attention_mask (torch.FloatTensor): Mask to avoid attention on padding tokens.
            token_type_ids (torch.LongTensor, optional): Segment IDs.
            position_ids (torch.LongTensor, optional): Token positions.
            head_mask (torch.FloatTensor, optional): Mask for attention heads.
            masked_lm_labels (torch.LongTensor, optional): Labels for MLM loss.

        Returns:
            tuple: If `masked_lm_labels` is provided, returns:
                - masked_lm_loss (torch.FloatTensor): Loss per sequence.
                - prediction_scores (torch.FloatTensor): Vocabulary prediction scores.
                - sequence_output (torch.FloatTensor): Final hidden states.
                - other outputs from the model.
        """
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
        return outputs

class Extractive:
    """ Class for performing extractive summarization based on masked LM scoring."""
    def __init__(self, tokenizer, model):
        """ Initialize tokenizer and model for extractive summarization.
        Args:
            tokenizer (str): Hugging Face tokenizer name.
            model (str): Hugging Face model name or path. 
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForMaskedLMwithLoss.from_pretrained(model).to(self.device).eval()

    def convert_data(self, inp_file, int_out_file):
        """Convert raw input JSON to intermediate format with references."""
        with open(int_out_file, 'w', encoding='utf-8') as fp:
            with open(inp_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
                for line in tqdm(lines):
                    try:
                        line = json.loads(line.strip())
                        title = line.get('title', '')
                        sections = line.get('sections', [])

                        for section in sections:
                            if section.get('references'):
                                refs = [t.replace('"', "'") for t in section['references']]
                                data_to_write = {
                                    'page_title': title,
                                    'section_title': section.get('title', ''),
                                    'content': section.get('content', ''),
                                    'references': refs,
                                    'domain': line.get('domain', 'unknown'),  # optional
                                    'src_lang': line.get('src_lang', 'unknown'),# optional
                                    'tgt_lang': line.get('tgt_lang', 'unknown')# optional
                                }
                                fp.write(json.dumps(data_to_write, ensure_ascii=False) + '\n')
                    except json.JSONDecodeError as e:
                        print(f"Skipping line due to error: {e}")

    def get_scores(self, splitted_sents, title, section_title):
        """Compute sentence scores using masked language model loss.
        Args:
            splitted_sents (list[str]): List of candidate sentences.
            title (str): Title of the document page.
            section_title (str): Title of the section.
        Returns:
            dict: Mapping of sentence -> negative loss score (higher is better).
        """
        scores = {}
        for sent in splitted_sents:
            st = self.tokenizer(section_title + ' ' + sent, return_tensors='pt', padding='max_length', truncation=True)
            input_ids, attention_mask = st['input_ids'].to(self.device), st['attention_mask'].to(self.device)
            res = self.model(input_ids, attention_mask, labels=input_ids)
            scores[sent] = -res[0].detach().cpu().numpy()

        return scores

    def get_top_k_sentences(self, scores_dict, k):
        """ Select top-k highest scoring sentences.
        Args:
            scores_dict (dict): Sentence -> score mapping.
            k (int): Number of sentences to return.
        Returns:
            list[str]: Top-k selected sentences.
        """
        scores_dict = sorted(scores_dict.items(), key = lambda x:-x[1])[:k]
        return [tup[0] for tup in scores_dict]

    def remove_bad_chars(self, text):
        """ Remove unwanted Unicode control/surrogate characters."""
        RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")
        return RE_BAD_CHARS.sub("", text)
    
    def perform_extractive_stage(self, inp_file, int_out_file, out_file, k):
        """ Perform full extractive summarization pipeline.
        Args:
            inp_file (str): Path to input JSON file.
            int_out_file (str): Path to intermediate JSON file (unused in this stage).
            out_file (str): Path to save final output file.
            k (int): Number of sentences to select per section.
        """
        print('==========Extracting Top K sentences==========')
        fp = open(out_file, 'w')
        with open(inp_file, 'r') as f:
            lines = f.readlines()
    
            for line in tqdm(lines):
                line = json.loads(line)
                title = line['title']
                for section in line['sections']:
                    section_title = section['title']
                    section_content = section['content']
                    refs = section['references']

                    if len(refs)>0:
                        splitted_sents = []
                        for ref in refs:
                            if ref != '':
                                ref = self.remove_bad_chars(ref)
                                try:
                                    lang = detect(ref)
                                    splitted_sents.extend(sentence_split(ref, lang=lang))
                                except:
                                    pass

                        if len(splitted_sents)>0:
                            scores = self.get_scores(splitted_sents, title, section_title)
                            top_k_sentences = self.get_top_k_sentences(scores, k)

                            output_entry = {
                                'page_title': title,
                                'section_title': section_title,
                                'content': section_content,
                                'references': top_k_sentences
                            }
                            # Add domain only if it exists in input
                            if 'domain' in line:
                                output_entry['domain'] = line['domain']
                            if 'src_lang' in line:
                                output_entry['src_lang'] = line['src_lang']
                            if 'tgt_lang' in line:
                                output_entry['tgt_lang'] = line['tgt_lang']
                                
                            fp.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
        fp.close()
        print('==========Done==========')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
    parser.add_argument('--inp_file', default=None, help='path to input json file for a given domain in given language')
    parser.add_argument('--int_out_file', default=None, help='path to intermediate output json file for a given domain in given language')
    parser.add_argument('--out_file', default=None, help='path to output json file for a given domain in given language')
    parser.add_argument('--tokenizer', default='xlm-roberta-base', help='which tokenizer to use')
    parser.add_argument('--model', default='xlm-roberta-base', help='which model to use')
    parser.add_argument('--top_k', default=50, type=int, help='how many sentences to pick')

    args = parser.parse_args()

    inp_file = args.inp_file
    int_out_file = args.int_out_file
    out_file = args.out_file
    k = args.top_k

    tokenizer = args.tokenizer
    model = args.model

    extractor = Extractive(tokenizer, model)

    extractor.perform_extractive_stage(inp_file, int_out_file, out_file, k)
