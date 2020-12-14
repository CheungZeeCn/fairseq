# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import logging

import torch

from fairseq import metrics, options, utils
from fairseq.criterions.refinement_nat_loss import RefinementLabelSmoothedDualImitationCriterion
from fairseq.data import LanguagePairDataset

from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.refinement import RefinementTask, load_tokenizer_plus_refinement_pair_dataset
from fairseq import utils
from fairseq.data import data_utils

# data part
from fairseq.data import TokenizerDictionary, Dictionary

logger = logging.getLogger(__name__)


@register_task('rebert')
class RefinementLevenshteinReBtpTask(RefinementTask):
    """
    在之前翻译任务的基础上，做纠错.
    相比lev-t 在encoder侧替换成了bert, 要指定bert预训练模型的路径
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        RefinementTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        parser.add_argument(
            "--load-source-middle",
            action="store_true",
            help="load source middle files for better training",
        )
        parser.add_argument('--load-hf-bert-from', type=str, default='',
                            help='load huggingface pretrained bert from path')

        parser.add_argument('--load-hf-bert-config-only', action='store_true',
                            help='only load config in the path so we can get a hf model')

        parser.add_argument(
            "--fix-bert-params",
            action="store_true",
            help='fix-bert-params'
        )

        parser.add_argument(
            "--share-bert-params",
            action="store_true",
            help='fix-bert-params'
        )

        parser.add_argument(
            "--pinyin-on",
            action="store_true",
            help='enable pinyin feature'
        )

        parser.add_argument(
            "--dual-policy-ratio", default=0.5, type=float, metavar='N',
            help='the probability of using dual policy in one pass of forward()'
        )
        parser.add_argument(
            "--middle-mode-ratio", default=0.5, type=float, metavar='N',
            help='the probability of using middle source data in one pass of forward()'
        )
        # parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
        #                     help='decoder embedding dimension')



    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """ 新的数据集格式，需要新的数据加载方法
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_tokenizer_plus_refinement_pair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_source_middle=self.args.load_source_middle,
            src_pinyin_dict=self.src_pinyin_dict,
            tgt_pinyin_dict=self.tgt_pinyin_dict
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                1,
                target_rank.masked_fill_(target_cutoff,
                                         max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args):
        """
            会在infer环节被使用 见 generate.py
        :param models:
        :param args:
        :return:
        """
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator_rbtp import IterativeRefinementGeneratorRbtp
        # infer 的时候， 调用这个对象的 generate
        return IterativeRefinementGeneratorRbtp(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False))

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError("Constrained decoding with the task is not supported")

        raise NotImplementedError("")
        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):

        model.train()
        sample['prev_target'] = self.inject_noise(sample['target'])

        if self.args.load_source_middle is True:
            assert isinstance(criterion, RefinementLabelSmoothedDualImitationCriterion)
            loss, sample_size, logging_output = criterion(model, sample, load_source_middle=True)
        else:
            loss, sample_size, logging_output = criterion(model, sample)

        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample['target'])
            loss, sample_size, logging_output = criterion(model, sample, load_source_middle=True)
        return loss, sample_size, logging_output

    # src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
    # 调用包装后的字典
    @classmethod
    def load_dictionary(cls, bert_pretrained_path):
        dictionary = TokenizerDictionary.load(bert_pretrained_path)
        return dictionary

    @classmethod
    def load_pinyin_dictionary(cls, file):
        dictionary = Dictionary.load(file)
        return dictionary

    # 要重新写 任务的初始化
    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # 这个task无论如何也是要指定原本预训练的model的路径才好初始化tokeniser的
        assert args.load_hf_bert_from != ''

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(args.load_hf_bert_from)
        tgt_dict = cls.load_dictionary(args.load_hf_bert_from)

        if args.pinyin_on is True:
            src_pinyin_dict = cls.load_pinyin_dictionary(os.path.join(paths[0], 'pinyin.dict'))
            tgt_pinyin_dict = cls.load_pinyin_dictionary(os.path.join(paths[0], 'pinyin.dict'))
        else:
            src_pinyin_dict = None
            tgt_pinyin_dict = None

        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        if src_pinyin_dict is not None:
            logger.info('pinyin ON, [{}] pinyin dictionary: {} types'.format(os.path.join(paths[0], 'pinyin.dict'),
                                                                             len(src_pinyin_dict)))
        else:
            logger.info('pinyin OFF')

        # 基类的这块的构造函数很简单，就是赋值了词典而已, 目前这个词典是被我们包起来的tokernizer
        return cls(args, src_dict, tgt_dict, src_pinyin_dict, tgt_pinyin_dict)
