# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset, # infer 的时候
    MiddleEnhancedTokenizerPlusLanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


# 针对bert引入的时候做的dataset
def load_tokenizer_plus_refinement_pair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine,
        dataset_impl, #upsample_primary,
        #left_pad_source,
        #left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        load_source_middle=True,
        src_pinyin_dict=None,
        tgt_pinyin_dict=None,
):
    """

    :param data_path:
    :param split: 类似 train， valid， test 这样的字符串
                     train.src-tgt.src         train.src-tgt.tgt
    :param src:
    :param src_dict:
    :param tgt:
    :param tgt_dict:
    :param combine:
    :param dataset_impl:
    :param upsample_primary:
    :param left_pad_source:
    :param left_pad_target:
    :param max_source_positions:
    :param max_target_positions:
    :param prepend_bos:
    :param load_alignments:
    :param truncate_source:
    :param append_source_id:
    :param num_buckets:
    :param shuffle:
    :param load_source_middle: 会有一些中间的diff 后的文件，打算用于加强训练;
    :return:
    """
    # 目前版本仅仅支持 raw_str dataset
    assert dataset_impl == 'raw_str'

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    src_mid_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        #src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        # raw_str data set here
        src_dataset = data_utils.load_indexed_raw_str_dataset(prefix + src)

        # 如果太长了，直接剪切掉
        if truncate_source:
            # raw_str 的操作就没有这么复杂，直接做字符串截断
            src_dataset = TruncateDataset(src_dataset, max_source_positions - 2)

        src_datasets.append(src_dataset)

        if load_source_middle == True:
            src_mid = src+'.middle'
            # 文件得在
            assert split_exists(split_k, src, tgt, src_mid, data_path)
            src_mid_dataset = data_utils.load_indexed_raw_str_dataset(prefix + src_mid)
            if truncate_source:
                src_mid_dataset = TruncateDataset(src_mid_dataset, max_target_positions - 2)
            src_mid_datasets.append(src_mid_dataset)
        else:
            src_mid_dataset = None

        tgt_dataset = data_utils.load_indexed_raw_str_dataset(prefix + tgt)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0
    assert len(src_datasets) == len(src_mid_datasets) or len(src_mid_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        src_mid_dataset = src_mid_datasets[0] if len(src_mid_datasets) > 0 else None
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        raise NotImplementedError("only len(src_datasets) == 1 supported plz merge them into one")

    if prepend_bos:
        raise NotImplementedError("prepend_bos not supported")

    # eos = None
    if append_source_id:
        raise NotImplementedError("append_source_id not supported")

    align_dataset = None
    if load_alignments:
        raise NotImplementedError("load_alignments=True is not supported now")

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    src_mid_dataset_sizes = src_mid_dataset.sizes if src_mid_dataset is not None else None
    # 支持mid是None的情况
    return MiddleEnhancedTokenizerPlusLanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        src_mid_dataset, src_mid_dataset_sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        # left_pad_source=left_pad_source,
        # left_pad_target=left_pad_target,
        # align_dataset=align_dataset,
        # eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        src_pinyin_dict=src_pinyin_dict,
        tgt_pinyin_dict=tgt_pinyin_dict
    )

# 针对bert引入的时候做的dataset
def load_tokenizer_refinement_pair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine,
        dataset_impl, #upsample_primary,
        #left_pad_source,
        #left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        load_source_middle=True
):
    """

    :param data_path:
    :param split: 类似 train， valid， test 这样的字符串
                     train.src-tgt.src         train.src-tgt.tgt
    :param src:
    :param src_dict:
    :param tgt:
    :param tgt_dict:
    :param combine:
    :param dataset_impl:
    :param upsample_primary:
    :param left_pad_source:
    :param left_pad_target:
    :param max_source_positions:
    :param max_target_positions:
    :param prepend_bos:
    :param load_alignments:
    :param truncate_source:
    :param append_source_id:
    :param num_buckets:
    :param shuffle:
    :param load_source_middle: 会有一些中间的diff 后的文件，打算用于加强训练;
    :return:
    """
    # 目前版本仅仅支持 raw_str dataset
    assert dataset_impl == 'raw_str'

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    src_mid_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        #src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        # raw_str data set here
        src_dataset = data_utils.load_indexed_raw_str_dataset(prefix + src)

        # 如果太长了，直接剪切掉
        if truncate_source:
            # raw_str 的操作就没有这么复杂，直接做字符串截断
            src_dataset = TruncateDataset(src_dataset, max_source_positions - 2)

        src_datasets.append(src_dataset)

        if load_source_middle == True:
            src_mid = src+'.middle'
            # 文件得在
            assert split_exists(split_k, src, tgt, src_mid, data_path)
            src_mid_dataset = data_utils.load_indexed_raw_str_dataset(prefix + src_mid)
            if truncate_source:
                src_mid_dataset = TruncateDataset(src_mid_dataset, max_target_positions - 2)
            src_mid_datasets.append(src_mid_dataset)
        else:
            src_mid_dataset = None

        tgt_dataset = data_utils.load_indexed_raw_str_dataset(prefix + tgt)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0
    assert len(src_datasets) == len(src_mid_datasets) or len(src_mid_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        src_mid_dataset = src_mid_datasets[0] if len(src_mid_datasets) > 0 else None
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        raise NotImplementedError("only len(src_datasets) == 1 supported plz merge them into one")

    if prepend_bos:
        raise NotImplementedError("prepend_bos not supported")

    # eos = None
    if append_source_id:
        raise NotImplementedError("append_source_id not supported")

    align_dataset = None
    if load_alignments:
        raise NotImplementedError("load_alignments=True is not supported now")

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    src_mid_dataset_sizes = src_mid_dataset.sizes if src_mid_dataset is not None else None
    # 支持mid是None的情况
    return MiddleEnhancedTokenizerLanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        src_mid_dataset, src_mid_dataset_sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        # left_pad_source=left_pad_source,
        # left_pad_target=left_pad_target,
        # align_dataset=align_dataset,
        # eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle
    )

@register_task('refinement')
class RefinementTask(FairseqTask):
    """
        从translationtask 转为 refinementtask


    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, src_pinyin_dict=None, tgt_pinyin_dict=None):
        """
            这个些dictionary 哪里来？
        :param args:
        :param src_dict:
        :param tgt_dict:
        """
        super().__init__(args)
        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_pinyin_dict = src_pinyin_dict
        self.tgt_pinyin_dict = tgt_pinyin_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        工厂模式啊，train处会调用这个， 然后才会把初始化好的词典喂进去, 如果我们要扩展pinyin，这里也应该加一个词典;

        todo: 要记得这里是加pinyin的入口

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        raise NotImplementedError()
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        raise NotImplementedError()
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        # self.datasets[split] = load_refinement_pair_dataset(
        #     data_path, split, src, self.src_dict, tgt, self.tgt_dict,
        #     combine=combine, dataset_impl=self.args.dataset_impl,
        #     upsample_primary=self.args.upsample_primary,
        #     left_pad_source=self.args.left_pad_source,
        #     left_pad_target=self.args.left_pad_target,
        #     max_source_positions=self.args.max_source_positions,
        #     max_target_positions=self.args.max_target_positions,
        #     load_alignments=self.args.load_alignments,
        #     truncate_source=self.args.truncate_source,
        #     num_buckets=self.args.num_batch_buckets,
        #     shuffle=(split != 'test'),
        #     load_source_middle=True
        # )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary,
                                   tgt_dict=self.target_dictionary,
                                   constraints=constraints)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s
        # 这块实际上是调用 generator.generate()
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
