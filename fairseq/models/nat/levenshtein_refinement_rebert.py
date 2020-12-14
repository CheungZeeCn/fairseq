# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture, FairseqEncoder

from pypinyin import pinyin, Style

from fairseq.models.transformer import (
    Embedding,
    TransformerDecoderLayer
)
import random

from fairseq.models.nat import (
    FairseqNATModel,
    # FairseqNATReBertDecoder,
    FairseqNATReBertPlusDecoder,
    FairseqNATDecoder,
    FairseqNATEncoder,
    ensemble_decoder
)

from fairseq.models.fairseq_encoder import EncoderOut

from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .levenshtein_utils import (
    _skip, _skip_encoder_out, _fill,
    _get_ins_targets, _get_del_targets, _get_del_target_by_middle,
    _apply_ins_masks, _apply_ins_words, _apply_del_words

)

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512

logger = logging.getLogger(__name__)


@register_model("levenshtein_refinement_rebert")
class LevenshteinRefinementBertTransformerPlusModel(FairseqNATModel):

    # 暂时不支持这种nat
    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="2,2,2",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help="share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action='store_true',
            help='instead of argmax, use sampling to predict the tokens'
        )
        parser.add_argument(
            "--share-bert",
            action="store_true",
            help="use the same bert for encoder and decoder",
        )
        # parser.add_argument(
        #     "--dual-policy-ratio", default=0.5, type=float, metavar='N',
        #     help='the probability of using dual policy in one pass of forward()'
        # )
        # parser.add_argument(
        #     "--middle-mode-ratio", default=0.5, type=float, metavar='N',
        #     help='the probability of using middle source data in one pass of forward()'
        # )
        parser.add_argument('--decoder-pinyin-embed-path', type=str, metavar='STR',
                            help='path to pre-trained pinyin decoder embedding')
        parser.add_argument('--encoder-pinyin-embed-path', type=str, metavar='STR',
                            help='path to pre-trained pinyin encoder embedding')
        # parser.add_argument('--load-hf-bert-from', type=str, default='',
        #                     help='load huggingface pretrained bert from path')

        # parser.add_argument('--load-hf-bert-config-only', action='store_true',
        #                    help='only load config in the path so we can get a hf model')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        levenshtein_rebert_base_architecture(args)

        # if args.encoder_layers_to_keep:
        #    args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        # if args.decoder_layers_to_keep:
        #    args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        # if args.share_all_embeddings:
        #     # 只能share pinyin的emb
        #     encoder_embed_tokens = cls.build_embedding(
        #         args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
        #     )
        #     raise NotImplementedError("等等，还不能share bert的encoder，待开发")
        # else:
        #     # tgt_dict 这里主要是给出emb的输入的维度, 这块在rebert是没用的
        #     decoder_embed_tokens = cls.build_embedding(
        #         args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        #     )
        src_pinyin_dict = task.src_pinyin_dict
        #tgt_pinyin_dict = task.tgt_pinyin_dict
        if args.pinyin_on is True:
            encoder_pinyin_embed_tokens = cls.build_embedding(
                args, src_pinyin_dict, args.encoder_pinyin_embed_dim, args.encoder_pinyin_embed_path
            )
            if args.encoder_pinyin_embed_path is None:
                encoder_pinyin_embed_tokens.apply(init_bert_params)
        else:
            encoder_pinyin_embed_tokens = None

        encoder = cls.build_encoder(args, src_dict, src_pinyin_dict, encoder_pinyin_embed_tokens)

        # decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        # 替换成encoder 直接给decoder 发挥类似 decoder_embed_tokens 的作用, 里面也有pinyin_embedding
        decoder = cls.build_decoder(args, tgt_dict, encoder)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, src_pinyin_dict, pinyin_embed_tokens):
        assert args.load_hf_bert_from != ''
        encoder = HuggingFaceBertPlusEncoder(args, src_dict, src_pinyin_dict, pinyin_embed_tokens)
        # if getattr(args, "apply_bert_init", False):
        #    encoder.apply(init_bert_params)
        return encoder

    """
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LevenshteinRefinementTransformerDecoder(args, tgt_dict, embed_tokens)
        #if getattr(args, "apply_bert_init", False):
        #    decoder.apply(init_bert_params)
        return decoder
    """

    @classmethod
    def build_decoder(cls, args, tgt_dict, encoder):
        decoder = LevenshteinRefinementPlusTransformerDecoder(args, tgt_dict, encoder)
        # if getattr(args, "apply_bert_init", False):
        #    decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def t2p(cls, t2p_buff, tokens_ids):
        l = tokens_ids.shape[1]
        pinyin_ids = t2p_buff.index_select(-1, tokens_ids.reshape(-1))
        pinyin_ids = pinyin_ids.reshape((-1, l)).contiguous()
        return pinyin_ids

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert tgt_tokens is not None, "forward function only supports training."

        # logging.info("self.args.load_source_middle: {}".format(self.args.load_source_middle))
        # logging.info("src_mid_tokens: {}".format(kwargs['src_mid_tokens']))
        # logging.info("src_mid_lengths: {}".format(kwargs['src_mid_lengths']))

        """
            想在训练这块做如下策略, 
                随机决定是否启用dual policy
                    一旦启用
                        从 prev 预测要增加几个，增加什么，然后用delete 来去在这个结果上进行学习
                    否则:     
                        从 delete这块就只能使用middle_src来进行学习
                随机决定是否使用middle_src
                    一旦启用
                        从middle_src 中学习 insert 内容，可以学习到正确答案的生成能力
                    如果不启用
                        在insert的学习中，可以来自random_delete策略, 强化生成能力
                        在不使用dual plicy的时候，也是要使用middle_src 来进行学习     
        """
        cut_off = random.random()
        dual_policy_ratio = getattr(self.args, 'dual_policy_ratio', 0.5)
        dual_policy_mode = False
        if cut_off <= dual_policy_ratio:
            dual_policy_mode = True

        # 由于dual_policy 不启用的使用必须要使用middle_source 来学习删除
        middle_mode = False
        middle_mode_ratio = getattr(self.args, 'middle_mode_ratio', 0.5)

        assert self.args.load_source_middle is True

        src_mid_tokens = kwargs['src_mid_tokens']
        src_mid_lengths = kwargs['src_mid_lengths']
        assert src_mid_tokens is not None, "when load_source_middle is true, we should get source middle dataset"
        # 多大的概率使用middle的数据来训练,
        # 如果mode不是middle就可以针对target 用random delete等等数据强化的方式(逻辑和原来的lev-t一样)，后续可以在这部分加入支持
        if cut_off <= middle_mode_ratio:
            middle_mode = True
        # 会影响 encoder 如果不删除
        del kwargs['src_mid_tokens']
        del kwargs['src_mid_lengths']

        """
            * 注意这里的改动 *
            本来, forward 函数关键输入是这样的: 
                src_tokens, src_lengths,prev_output_tokens, tgt_tokens
                关于上面提及的prev_output_tokens, 在训练阶段是来自sample['prev_target'], 默认是random delete后的字符串, 参考
                nat_loss/refinement_nat_loss forward() 中的实现.
            我们在kwargs 引入了事先计算好的 src_mid_tokens, src_mid_lengths, 代表 中间输入, 
            来自从source中删除对齐后和target字符串不一样的地方。
            
            1.原来的训练有三个主要步骤: 
                a. forward_mask_ins, 针对被random delete(默认操作，可以命令行改) 后生成的prev_output_tokens, 我们去猜insert的个数和位置
                b. forward_word_ins, 针对计算好的masked_tgt_tokens, 我们去猜对应的地方应该插入哪些token
                c. forward_word_del, 针对上一步预测出来的词，和target进行对比，然后训练应该delete什么词
            2.原生的过程是针对翻译和生成的，但是针对原文增强类任务，我们引入了对齐后的中间diff结果，可以做哪些训练过程的增强？
            我们希望训练过程中可以更加有针对性(但是在原生的模型上，可能会影响语言模型的学习,估计效果会下降，不过训练次数提升，
            或者使用预训练encoder估计会有提升)
                a. 我们想训练的过程中有一定的概率更加关注，错误点,相当于我们的中间结果明确标注出来了，应该删除的地方;
                b. 利用中间结果，是可以更好的去预测应该插入的个数和地方，相当于加强了forward_mask_ins和forward_word_ins 
                    b1. forward_mask_ins(prev_output_tokens=source_middle) 
                    b2. forward_word_ins(prev_output_tokens=source_middle和tgt_token计算得出的masked_tgt_tokens) 
                c. 利用中间结果，我们可以更好的去预测delete
                    c1. 按照原来的步骤预测出来和target不一致的，就应该delete; (这个跟原来的一样)
                    c2. 如果可以利用中间结果和原来的输入做一个对比，也可以标记出原始的输入多出来的是哪些，然后那些是应该删除的(利用libnat)
            3. 同样的，有针对的训练过后，对应的decoder在生成的时候，prev可以是原来的字符串, 这样也要求decoder要有足够锐利的眼光去识别prev中的错误信息;
        """

        # encoding
        # encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # logging.info("in forward src_tokens: {}".format(src_tokens))
        encoder_out = self.encoder(src_tokens)
        # logging.info("encoder_out {}".format(encoder_out))

        # generate training labels for insertion
        # validating :
        if self.training is False:
            # 不是training 的话不应该进入这个环节, 不过training 的 valid的时候可以进来哦
            # assert self.training is True, "should not be here in this forward() when you are not training the model"
            # 如果不是训练，就应该按照gold label 来计算, 所以
            # logger.info("self.training is False")
            middle_mode = True
            dual_policy_mode = False

        # logger.info(
        #     "forward policies: dual_policy_mode:{} {} middle_mode:{} {}".format(dual_policy_mode, dual_policy_ratio,
        #                                                                         middle_mode, middle_mode_ratio))

        # prev_output_tokens = src_mid_tokens
        ori_prev_output_tokens = prev_output_tokens

        # 训练insert的时候，使用gold label
        if middle_mode is True:
            prev_output_tokens = src_mid_tokens
            masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
                # prev_output_tokens, tgt_tokens, self.pad, self.unk
                prev_output_tokens, tgt_tokens, self.pad, self.tgt_dict.tokenizer.mask_token_id
            )
        else:
            # 使用random_delete 等noise
            masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
                # prev_output_tokens, tgt_tokens, self.pad, self.unk
                prev_output_tokens, tgt_tokens, self.pad, self.tgt_dict.tokenizer.mask_token_id
            )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        # logging.info(f"prev_output_tokens: {prev_output_tokens.shape} {prev_output_tokens}")

        prev_output_tokens_plus = self.tgt_dict.add_batch_plus(prev_output_tokens)
        masked_tgt_tokens_plus = self.tgt_dict.add_batch_plus(masked_tgt_tokens)
        if self.args.pinyin_on is True:
            prev_output_tokens_pinyin = self.t2p(self.encoder.t2p_buff, prev_output_tokens)
            prev_output_tokens_plus['pinyin'] = prev_output_tokens_pinyin
            masked_tgt_tokens_plus['pinyin'] = self.t2p(self.encoder.t2p_buff, masked_tgt_tokens)
        # logging.info(f"prev_output_tokens_plus: {prev_output_tokens_plus}")
        # raise NotImplementedError()

        # insert how many
        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            # prev_output_tokens=prev_output_tokens,
            prev_output_tokens_plus=prev_output_tokens_plus,
            # prev_output_tokens_pinyin=prev_output_tokens_pinyin,
            encoder_out=encoder_out
        )
        # insert what
        # logger.info(f"src_tokens['input_ids']: {src_tokens['input_ids']}")
        # logger.info(f"prev_output_tokens: {prev_output_tokens}")
        # logger.info(f"masked_tgt_tokens: {masked_tgt_tokens}")
        # logger.info(f"masked_tgt_masks: {masked_tgt_masks}")
        # logger.info(f"mask_ins_out: {mask_ins_out}")
        # logger.info(f"mask_ins_targets: {mask_ins_targets}")
        # raise NotImplementedError()

        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens_plus=masked_tgt_tokens_plus,
            # prev_output_tokens_pinyin=masked_tgt_tokens_pinyin,
            encoder_out=encoder_out
        )

        # make online prediction #默认是False
        if self.decoder.sampling_for_deletion:
            raise NotImplementedError()
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1).view(
                word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        # generate training labels for deletion
        if dual_policy_mode is True:  # 默认方法, 训练delete
            # raise NotImplementedError("要待后续开发")
            # 计算两者不一样的, 标记为要delete
            word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
            # 加持
            # 还得加拼音
            word_predictions_plus = self.tgt_dict.add_batch_plus(word_predictions)
            if self.args.pinyin_on is True:
                word_predictions_plus['pinyin'] = self.t2p(self.encoder.t2p_buff, word_predictions)
            word_del_out, _ = self.decoder.forward_word_del(
                normalize=False,
                # prev_output_tokens=word_predictions,
                prev_output_tokens_plus=word_predictions_plus,
                encoder_out=encoder_out)
            word_del_masks = word_predictions.ne(self.pad)
        else:  # 用middle的时候 相当于一个错误识别模块
            word_del_targets = _get_del_target_by_middle(src_mid_tokens, src_tokens['input_ids'], self.pad)
            word_del_out, _ = self.decoder.forward_word_del(
                normalize=False,
                prev_output_tokens_plus=src_tokens,
                # prev_output_tokens_pinyin=src_tokens['pinyin'],
                encoder_out=encoder_out)
            word_del_masks = src_tokens['input_ids'].ne(self.pad)
            # logging.info("word_del_targets: {}".format(word_del_targets))
            # raise NotImplementedError("zz")

        return {
            "mask_ins": {
                "out": mask_ins_out, "tgt": mask_ins_targets,
                "mask": mask_ins_masks, "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": masked_tgt_masks, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "word_del": {
                "out": word_del_out, "tgt": word_del_targets,
                "mask": word_del_masks
            }
        }

    def forward_decoder(
            self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        # logger.info(f"output_tokens: {output_tokens}")
        # logger.info(f"output_scores: {output_scores}")

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        output_tokens_plus = self.tgt_dict.add_batch_plus(output_tokens)
        if self.args.pinyin_on is True:
            output_tokens_plus['pinyin'] = self.t2p(self.encoder.t2p_buff, output_tokens)
        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        # logger.info("can_del_word {} {}".format(can_del_word.shape, can_del_word))
        if can_del_word.sum() != 0:  # we cannot delete, skip
            # word_del_score, word_del_attn = self.decoder.forward_word_del(
            #     normalize=True,
            #     prev_output_tokens_plus=_skip(output_tokens, can_del_word),
            #     encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word)
            # )
            word_del_score, word_del_attn = self.decoder.forward_word_del(
                normalize=True,
                prev_output_tokens_plus=_skip(output_tokens_plus, can_del_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word)
            )
            word_del_pred = word_del_score.max(-1)[1].bool()
            # logger.info(word_del_score)
            # logger.info(word_del_pred)
            # raise NotImplementedError()

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        output_tokens_plus = self.tgt_dict.add_batch_plus(output_tokens)
        if self.args.pinyin_on is True:
            output_tokens_plus['pinyin'] = self.t2p(self.encoder.t2p_buff, output_tokens)

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            # mask_ins_score, _ = self.decoder.forward_mask_ins(
            #     normalize=True,
            #     prev_output_tokens=_skip(output_tokens, can_ins_mask),
            #     encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask)
            # )
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens_plus=_skip(output_tokens_plus, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask)
            )
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        output_tokens_plus = self.tgt_dict.add_batch_plus(output_tokens)
        if self.args.pinyin_on is True:
            output_tokens_plus['pinyin'] = self.t2p(self.encoder.t2p_buff, output_tokens)

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            # word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
            #     normalize=True,
            #     prev_output_tokens=_skip(output_tokens, can_ins_word),
            #     encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_word)
            # )
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens_plus=_skip(output_tokens_plus, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_word)
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        """
            在生成的时候会被调用来初始化原始的输出token；
            todo: 对参数敏感的，使用 src_tokens 来进行初始化, 但是这个scores就比较尴尬
                怎么取值合适, 这个模型的后续步骤有用到这个scores吗？
        :param encoder_out:
        :param src_tokens:
        :return:
        """
        """
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens[:, 1] = self.eos
        """
        # 将src_tokens 作为原始的输出, 这样模型的下一步工作就是要识别src_token哪里有问题
        initial_output_tokens = src_tokens['input_ids'].detach().clone()
        # logging.info("copy src for init here")

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )


# class HuggingFaceBertEncoder(FairseqEncoder):
class HuggingFaceBertPlusEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, pinyin_dict=None, pinyin_embed_tokens=None):
        super().__init__(dictionary)
        try:
            from transformers import BertModel, BertTokenizer, BertConfig
        except ImportError:
            raise ImportError(
                '\n\nPlease install huggingface/transformers with:'
                '\n\n  pip install transformers'
                '\n\nOr to make local edits, install the submodule:'
                '\n\n  git submodule update --init '
                'fairseq/models/huggingface/transformers'
            )

        # logging.info(args)
        # raise NotImplementedError(args.load_hf_bert_from)
        load_hf_bert_from = getattr(args, 'load_hf_bert_from', '')
        assert load_hf_bert_from != ''
        model_path = load_hf_bert_from

        config = BertConfig.from_pretrained(model_path)

        # logging.info("args: {}".format(args))
        if getattr(args, 'load_hf_bert_config_only', False) is True:
            logger.info(
                "now we will init the hf_bert model from config without the weights,"
                " since we will restore the weights later")
            self.model = BertModel(config)
        else:
            logger.info("now we will init the hf_bert model from {} with all the weights".format(model_path))
            self.model = BertModel.from_pretrained(model_path)
        # logging.info("DEBUG: after loading hf_bert: encoder.layer.11.output.dense.weight[0][10]{}".format(
        #     self.model.state_dict()['encoder.layer.11.output.dense.weight'][0][:10]))
        # self.model = self.model
        if args.fix_bert_params is True:
            for p in self.model.parameters():
                p.requires_grad = False
        self.tokenizer = dictionary.tokenizer
        self.dictionary = dictionary
        self.args = args
        self.config = config
        # self.model.embeddings
        # could be None
        self.pinyin_dict = pinyin_dict
        self.pinyin_embed_tokens = pinyin_embed_tokens

        if args.pinyin_on is True:
            t2p_buff = self.build_token_ids_to_pinyin_ids_buff(self.tokenizer, self.pinyin_dict)
            self.register_buffer('t2p_buff', t2p_buff)
            logging.info("t2p_buff.shape:{} self.t2p_buff:{}".format(self.t2p_buff.shape, self.t2p_buff))
        else:
            t2p_buff = None
            self.register_buffer('t2p_buff', t2p_buff)
            logging.info("t2p_buff is None")

        # raise NotImplementedError()

    # build tokenid2pinyinid buffer
    @classmethod
    def build_token_ids_to_pinyin_ids_buff(cls, tokenizer, pinyin_dict):
        py_vocab = [x[0] for x in pinyin(tokenizer.vocab, style=Style.NORMAL)]
        py_ids = []
        for i_pinyin in py_vocab:
            if i_pinyin == '[PAD]':
                py_idx = pinyin_dict.pad()
            elif i_pinyin == '[CLS]':
                py_idx = pinyin_dict.bos()
            elif i_pinyin == '[UNK]':
                py_idx = pinyin_dict.unk()
            elif i_pinyin == '[SEP]':
                py_idx = pinyin_dict.eos()
            else:
                py_idx = pinyin_dict.index(i_pinyin)
            py_ids.append(py_idx)
        t2p_buff = torch.tensor(py_ids, requires_grad=False).long()
        return t2p_buff

    # 为了和一些decoder的基类代码兼容，加的
    @property
    def embedding(self):
        return self.model.embeddings.word_embeddings

    # def reorder_encoder_out(self, ):
    #    super().reorder_encoder_out()
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def forward(self, src_tokens, return_all_hiddens=False, return_pool=False):
        """
        """
        # logger.info(src_tokens)
        # raise NotImplementedError("xx")
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens, return_pool=return_pool)
        #
        # logger.info(src_tokens)
        # logger.info(x)

        encoder_padding_mask = src_tokens['input_ids'].eq(self.dictionary.pad())

        x = x.transpose(0, 1).contiguous()
        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def extract_features(self, src_tokens, return_all_hiddens=False, return_pool=False, **unused):
        # inner_states, _ = self.model(**src_tokens, output_hidden_states=not return_all_hiddens)
        # logger.info(src_tokens)
        bert_input = {}
        for k in ('input_ids', 'token_type_ids', 'attention_mask'):
            bert_input[k] = src_tokens[k]
        inner_states = self.model(**bert_input, output_hidden_states=return_all_hiddens)
        # logger.info(self.pinyin_embed_tokens, len(self.pinyin_dict.symbols))
        # pinyin_input = src_tokens['pinyin']
        # logger.info(pinyin_input)
        # raise NotImplementedError()
        # raise NotImplementedError("")
        # 可以的，还是在cuda
        # logger.info(inner_states)
        # raise NotImplementedError("")
        # 转 float 32 不然好像会自动变half
        bert_features = inner_states[0]
        # 这里面加入pinyin feature
        if self.args.pinyin_on is True:
            pinyin_input = src_tokens['pinyin']
            embed_out = self.pinyin_embed_tokens(pinyin_input)
            features = torch.cat([bert_features, embed_out], axis=-1)
        else:
            features = bert_features

        # logging.info("bert_features.shape {}".format(bert_features.shape))
        # logging.info("embed_out.shape {} embed_out {}".format(embed_out.shape, embed_out))

        # logger.info(
        #     "bert_features.shape: {} ; embout.shape: {}, bert_features.shape:{}".format(bert_features.shape,
        #                                                                                 embed_out.shape,
        #                                                                                 features.shape))
        # raise NotImplementedError()
        return features, {'inner_states': inner_states[2] if return_all_hiddens else None,
                          'pool': inner_states[1] if return_pool else None,
                          'bert_features': bert_features}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.model.config.max_position_embeddings


class LevenshteinRefinementPlusTransformerDecoder(FairseqNATReBertPlusDecoder):
    """
        可以考虑使用encoder中的bert来替换掉emb层？ 这块涉及要替换的父类层数有点多,所以我直接替换了两层父类
    """

    def __init__(self, args, dictionary, encoder: HuggingFaceBertPlusEncoder, no_encoder_attn=False):
        # 这个应该会默认的 build self.layers, plus 了以后，也要处理一下pinyin才行哦
        super().__init__(
            args, dictionary, encoder, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        # 256 是因为最多只能ins 256 的长度吧？这块实际上应该还能调整一下，如果是不需要完全的生成原文
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(',')]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(self.early_exit[1])
            ])
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(self.early_exit[0])
            ])

        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(args, "no_share_discriminator", False), "must set saperate discriminator"
            self.layers_msk = self.layers_del

        # 默认情况下，layers_msk 和 layers_del 都是 None ?
        self.encoder = encoder

        self.pinyin_embed_tokens = encoder.pinyin_embed_tokens

        # self.register_buffer('t2p_buff', self.encoder.t2p_buff)
        self.t2p_buff = self.encoder.t2p_buff

        if args.share_bert is False:
            # 暂时先不写这块
            raise NotImplementedError()
        else:
            self.model = self.encoder.model


    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.encoder.max_positions()

    def _extract_features(self, prev_tokens, return_all_hiddens=False, return_pool=False, **unused):
        bert_input = {}
        for k in ('input_ids', 'token_type_ids', 'attention_mask'):
            bert_input[k] = prev_tokens[k]
        inner_states = self.model(**bert_input, output_hidden_states=return_all_hiddens)
        bert_features = inner_states[0]
        # 这里面加入pinyin feature
        if self.args.pinyin_on is True:
            pinyin_input = prev_tokens['pinyin']
            embed_out = self.pinyin_embed_tokens(pinyin_input)
            features = torch.cat([bert_features, embed_out], axis=-1)
        else:
            features = bert_features

        return features, {'inner_states': inner_states[2] if return_all_hiddens else None,
                          'pool': inner_states[1] if return_pool else None,
                          'bert_features': bert_features}

    def extract_features(
            #self, prev_output_tokens_plus, encoder_out=None, early_exit=None, layers=None,
            self, prev_output_tokens_plus, encoder_out, early_exit=None, layers=None,
            **unused
    ):
        prev_output_tokens = prev_output_tokens_plus['input_ids']
        x, _ = self._extract_features(prev_output_tokens_plus)

        # B x T x C -> T x B x C
        # logger.info("shape x.shape {}, and pad  {}".format(x.shape, self.padding_idx))
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        # done: 检查下这个 layer 什么时候知道 encoder_out 的维度信息? 在MHA的实现处，使用了 encoder_embed_dim, cool
        # logger.info(layers[0])
        for _, layer in enumerate(layers[: early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # return x, {"attn": attn, "inner_states": inner_states}
        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens_plus, **unused):
        #prev_output_tokens = prev_output_tokens_plus['input_ids']
        features, extra = self.extract_features(
            prev_output_tokens_plus, encoder_out=encoder_out, early_exit=self.early_exit[1],
            layers=self.layers_msk,
            **unused
        )
        # logger.info(f"features.shape {features.shape}")
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        # logger.info(f"features_cat.shape {features_cat.shape}")
        # logger.info(f"self.embed_mask_ins.weight.shape {self.embed_mask_ins.weight.shape}")
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens_plus, **unused):
        features, extra = self.extract_features(
            prev_output_tokens_plus, encoder_out=encoder_out, early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_del(self, normalize, encoder_out, prev_output_tokens_plus, **unused):
        features, extra = self.extract_features(
            prev_output_tokens_plus, encoder_out=encoder_out, early_exit=self.early_exit[0],
            layers=self.layers_del,
            **unused
        )
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']


@register_model_architecture("levenshtein_refinement_rebert", "levenshtein_refinement_rebert")
def levenshtein_rebert_base_architecture(args):
    # about pinyin
    args.pinyin_on = getattr(args, "pinyin_on", False)
    args.pinyin_embed_path = getattr(args, "pinyin_embed_path", None)
    args.pinyin_embed_dim = getattr(args, "pinyin_embed_dim", 8)

    args.encoder_pinyin_embed_path = getattr(args, "encoder_pinyin_embed_path", None)
    args.encoder_pinyin_embed_dim = getattr(args, "encoder_pinyin_embed_dim", 8)
    if args.encoder_pinyin_embed_path is None:
        args.encoder_pinyin_embed_path = args.pinyin_embed_path
        args.encoder_pinyin_embed_dim = args.pinyin_embed_dim

    args.decoder_pinyin_embed_path = getattr(args, "decoder_pinyin_embed_path", None)
    args.decoder_pinyin_embed_dim = getattr(args, "decoder_pinyin_embed_dim", 8)
    if args.decoder_pinyin_embed_path is None:
        args.decoder_pinyin_embed_path = args.encoder_pinyin_embed_path
        args.decoder_pinyin_embed_dim = args.encoder_pinyin_embed_dim
    # about pinyin done

    if args.pinyin_on is True:
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768 + args.encoder_pinyin_embed_dim)
        args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768 + args.decoder_pinyin_embed_dim)
    else:
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
        args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    # args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    # 为了兼容底层的transformer
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_bert = getattr(args, "share_bert", True)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    # bert 预训练+微调 or 特征提取
    args.fix_bert_params = getattr(args, "fix_bert_params", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    # todo ?? decoder_input_dim ? 就是input_dim 可以和embed_dim 不一样，这块就可以自动加一层转换
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)


@register_model_architecture(
    "levenshtein_refinement_rebert", "levenshtein_refinement_rebert_decoder_2layers"
)
def levenshtein_refinement_rebert_decoder_2layers(args):
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    levenshtein_rebert_base_architecture(args)


@register_model_architecture(
    "levenshtein_refinement_rebert", "levenshtein_refinement_rebert_decoder_6layers"
)
def levenshtein_refinement_rebert_decoder_6layers(args):
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    levenshtein_rebert_base_architecture(args)


@register_model_architecture(
    "levenshtein_refinement_rebert", "levenshtein_refinement_rebert_decoder_12layers"
)
def levenshtein_refinement_rebert_decoder_12layers(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    levenshtein_rebert_base_architecture(args)
