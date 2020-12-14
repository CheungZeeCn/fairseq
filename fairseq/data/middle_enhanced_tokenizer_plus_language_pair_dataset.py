# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import pypinyin
from pypinyin import pinyin, Style

from fairseq.data import data_utils, FairseqDataset

from fairseq.data.dictionary import Dictionary

logger = logging.getLogger(__name__)


def collate(
        samples,
        src_dict,
        tgt_dict,
        pad_idx,
        eos_idx,
        # left_pad_source=True,
        # left_pad_target=False,
        input_feeding=True,
        pad_to_length=None,
        src_pinyin_dict=None,
        tgt_pinyin_dict=None
):
    if len(samples) == 0:
        return {}
    # logger.info("before collate samples {}".format(samples))
    # raise NotImplementedError()
    """
        输入的样子
        example = {
            'id': index,
            'source': src_item, # 原始字符串
            'source_middle': src_mid_item, # 原始字符串
            'target': tgt_item, # 原始字符串
        }
    """
    assert pad_to_length == None, "pad_to_length not supported"
    if src_pinyin_dict is None:
        pinyin_on = False
    else:
        pinyin_on = True

    # logger.info("in collate sampels: {}".format(samples))
    def merge_str(key, dictionary, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [dictionary.encode_line(s[key]).long() for s in samples],
            pad_idx, eos_idx, False, move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    def merge_pinyin_str(key, tokenizer, dictionary, move_eos_to_beginning=False, pad_to_length=None, append_bos=True):
        batch_pinyin_str = [get_pinyin(s[key], tokenizer, dictionary) for s in samples]
        bos = dictionary.bos()
        ret_tensor = data_utils.collate_tokens(
            [dictionary.encode_line(s).long() if append_bos is False else torch.cat(
                [torch.LongTensor([bos]), dictionary.encode_line(s).long()]) for s in batch_pinyin_str],
            dictionary.pad(), dictionary.eos(), False, move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )
        return ret_tensor

    def merge(key, left_pad=False, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    def merge_from_str(key, tokenizer, pinyin_dict=None):
        """
        return examples like:
        {'input_ids': tensor([[101, 2644, 1962, 102, 0],
                              [101, 2769, 738, 1962, 102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0],
                                                                                       [0, 0, 0, 0, 0]]),
         'attention_mask': tensor([[1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1]])
        'pinyin': ...
                                   }
        """
        ret = dict(tokenizer([s[key] for s in samples], padding=True, return_tensors='pt'))
        # logger.info("f{pinyin_dict}")
        # logger.info(dir(pinyin_dict))
        # 拼音字符串列表
        if pinyin_dict is not None:
            batch_pinyin_encoded = merge_pinyin_str(key, tokenizer, pinyin_dict, move_eos_to_beginning=False,
                                                    pad_to_length=pad_to_length)
            ret['pinyin'] = batch_pinyin_encoded
        return ret

    def get_pinyin(text, tokenizer, pinyin_dict):
        t_list = tokenizer.tokenize(text)
        to_strs = []
        t_list = [t[-1] for t in t_list]
        try:
            for i, py_sub_list in enumerate(pinyin(t_list, style=Style.NORMAL)):
                # logger.info(py_sub_list)
                i_pinyin = py_sub_list[0]
                if i_pinyin == t_list[i]:
                    i_pinyin = pinyin_dict.unk_word
                to_strs.append(i_pinyin)
            ret_pinyin_str = " ".join(to_strs)
        except Exception as e:
            logger.info(e)
            logger.info(text)
            logger.info(t_list)
            logger.info(pinyin(t_list, style=Style.NORMAL))
            logger.info(i)
            raise e
        # logger.info(ret_pinyin_str)
        return ret_pinyin_str

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokened = merge_from_str('source', src_dict.tokenizer, src_pinyin_dict)
    # sort by descending source length
    pad_idx = src_dict.pad_index
    src_lengths = src_tokened['input_ids'].ne(pad_idx).sum(dim=1)
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    for k, v in src_tokened.items():
        src_tokened[k] = v.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_str('target', tgt_dict)
        target = target.index_select(0, sort_order)
        tgt_lengths = target.ne(pad_idx).sum(dim=1)
        ntokens = tgt_lengths.sum().item()
        if pinyin_on is True:
            tgt_pinyins = merge_pinyin_str('target', tgt_dict.tokenizer, tgt_pinyin_dict, move_eos_to_beginning=False,
                                           pad_to_length=pad_to_length)
            tgt_pinyins = tgt_pinyins.index_select(0, sort_order)
        else:
            tgt_pinyins = None

        if samples[0].get('prev_output_tokens', None) is not None:
            # 什么时候会进来这里呢, infer 的时候？ 进来的话，这里不能是str
            raise NotImplementedError("check here")
            prev_output_tokens = merge('prev_output_tokens', left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge_str('target', tgt_dict, move_eos_to_beginning=True)
            if pinyin_on is True:
                prev_output_tokens_pinyins = merge_pinyin_str('target', tgt_dict.tokenizer, tgt_pinyin_dict,
                                                              move_eos_to_beginning=True,
                                                              pad_to_length=pad_to_length)
                prev_output_tokens_pinyins = prev_output_tokens_pinyins.index_select(0, sort_order)
            else:
                prev_output_tokens_pinyins = None
    else:
        ntokens = src_lengths.sum().item()
    # added for source middle
    if samples[0].get('source_middle', None) is not None:
        src_mid_tokens = merge_str('source_middle', src_dict)
        src_mid_tokens = src_mid_tokens.index_select(0, sort_order)
        src_mid_lengths = src_mid_tokens.ne(pad_idx).sum(dim=1)
        if pinyin_on is True:
            src_mid_pinyins = merge_pinyin_str('source_middle', src_dict.tokenizer, src_pinyin_dict,
                                           move_eos_to_beginning=False,
                                           pad_to_length=pad_to_length).index_select(0, sort_order)
        else:
            src_mid_pinyins = None
    else:
        src_mid_tokens = None
        src_mid_lengths = None
        src_mid_pinyins = None

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokened,
            'src_lengths': src_lengths,
            'src_mid_tokens': src_mid_tokens,
            'src_mid_lengths': src_mid_lengths,
            'src_mid_pinyins': src_mid_pinyins
        },
        'target': target,
        # 其实这个没啥用，就是放这里看看做debug
        'target_pinyins': tgt_pinyins,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens.index_select(0, sort_order)
        if pinyin_on is True:
            batch['net_input']['prev_output_tokens_pinyins'] = prev_output_tokens_pinyins.index_select(0, sort_order)
        else:
            batch['net_input']['prev_output_tokens_pinyins'] = None

    # logger.info("after collate collate samples {}".format(batch))
    return batch


class MiddleEnhancedTokenizerPlusLanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.
    Args:

    """

    def __init__(
            self, src, src_sizes, src_dict,
            src_mid=None, src_mid_sizes=None, src_mid_dict=None,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            shuffle=True, input_feeding=True,
            # remove_eos_from_source=False, append_eos_to_target=False,
            # align_dataset=None,
            # constraints=None,
            # append_bos=False, eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            src_pinyin_dict=None,
            tgt_pinyin_dict=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(tgt), "Source and target must contain the same number of examples"
        self.src = src
        self.src_mid = src_mid
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.src_mid_sizes = np.array(src_mid_sizes) if src_mid_sizes is not None else None
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.src_mid_dict = src_mid_dict
        self.tgt_dict = tgt_dict
        # todo: 这两个按道理应该不影响后续逻辑，mark一下，待确认
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        # ---
        self.shuffle = shuffle
        # todo: mark 这块应该是 teacher forcing 的开关
        self.input_feeding = input_feeding
        # 直接注释掉先
        # self.remove_eos_from_source = remove_eos_from_source
        # self.append_eos_to_target = append_eos_to_target
        # self.align_dataset = align_dataset
        # if self.align_dataset is not None:
        #     assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        # self.constraints = constraints
        # self.append_bos = append_bos
        # self.eos = (eos if eos is not None else src_dict.eos())
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            # 这块就是不能够支持呀，因为要提前pad，如果要支持，就是要先把tokenizer的各个子步骤拆解先
            raise NotImplementedError("num_buckets > 0 Not Supported Yet")
        self.buckets = None
        self.src_pinyin_dict = src_pinyin_dict
        self.tgt_pinyin_dict = tgt_pinyin_dict

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_mid_item = self.src_mid[index] if self.src_mid is not None else None
        src_item = self.src[index]
        # 如果后续要加pinyin，就在这里加，类似source_pinyin 这样加入进来
        example = {
            'id': index,
            'source': src_item,
            'source_middle': src_mid_item,
            'target': tgt_item,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            self.src_dict,
            self.tgt_dict,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            # left_pad_source=self.left_pad_source,
            # left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            src_pinyin_dict=self.src_pinyin_dict,
            tgt_pinyin_dict=self.tgt_pinyin_dict
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res['net_input']['src_tokens']
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res['net_input']['src_lang_id'] = torch.LongTensor(
                    [[self.src_lang_id]]
                ).expand(bsz, 1).to(src_tokens)
            if self.tgt_lang_id is not None:
                res['tgt_lang_id'] = torch.LongTensor(
                    [[self.tgt_lang_id]]
                ).expand(bsz, 1).to(src_tokens)
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[
                    np.argsort(self.tgt_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            raise NotImplementedError("buckets not supported")
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')
            ]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.src_mid is not None:
            self.src_mid.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """ Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if max_sizes is None:
            return indices, []
        if type(max_sizes) in (int, float):
            max_src_size, max_tgt_size = max_sizes, max_sizes
        else:
            max_src_size, max_tgt_size = max_sizes
        if self.tgt_sizes is None:
            ignored = indices[self.src_sizes[indices] > max_src_size]
        else:
            ignored = indices[(self.src_sizes[indices] > max_src_size) |
                              (self.tgt_sizes[indices] > max_tgt_size)]
        if len(ignored) > 0:
            if self.tgt_sizes is None:
                indices = indices[self.src_sizes[indices] <= max_src_size]
            else:
                indices = indices[(self.src_sizes[indices] <= max_src_size) &
                                  (self.tgt_sizes[indices] <= max_tgt_size)]
        return indices, ignored.tolist()
