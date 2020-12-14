# rebert: 将bert集成到 levenshtein-transformer 的框架中，并尝试用于纠错 #

--------------------------------------------------------------------------------

## 先直接上几张图作为引子吧 ##


1. 当前进展:
   1. 端到端纠错任务比想象中难，delete loss 比较容易下降，但是生成部分的loss 当前还没有办法降下去；目前我手头还没有资源进行大规模调参。
   1. 可以轻巧的用于data2text任务，生成模板的连接词, 参考
   1. 之前是基于 0.9.0 的fairseq写的，发现上了0.10.0 了，重新摘出来了一下, 框架底层有些不一样，有些代码还没来得及补充完, 不过训练是没问题的
1. 训练模式的支持和开关说明：
   1. 是否启用拼音--pinyin-on
      1. 启用的话就构造t2p缓存(decoder 拿到token id 可以直接映射 到 pinyin id, 然后再直接取出pinyin emb, 这种做法性能会很好，但是不支持多音字)
      1. todo: 指定的pinyin emb 路径, 当前是没有支持训练好pinyin emb的, 从0开始训练;
   1. 关于bert
      1. --fix-bert-param 是否固定bert 的参数， 固定 bert 参数其实就是把bert用作特征提取了，如果没有固定，就是 预训练+微调的模式
      1. todo: --share-bert 目前默认share-bert 不share 的情况还没有实现
   1. 训练模式
      1. --dual-policy-ratio 多大的概率使用dual policy 来训练（也就是训练delete的时候，基于之前insert的结果，如果insert错了，delete可以指出来） 
      1. —-middle-ratio 多大的概率用 middle target/source 来引导训练；这个middle 是做数据处理的时候用difflib来生成的，相当于理想情况下删除了source 串中错误
      字符后的字符串，针对删除操作，这就是golden label， 针对插入操作，也是作为一个prev_token 来有针对性的训插入任务。
1. 怎么训练
   1. 样例数据集，空间所限，放了个比较难的任务但是train/valid/test 是一样的数据集做实验
   1. 首先你要参考pull 这份代码，然后参考fairseq官方文档，完成安装;
   1. 输入
   ```bash
    $ cd [FAIRSEQ_DIR] && mkdir -p  outputs/CGEC_debug_2.pinyinoff.20201214
    $ CUDA_VISIBLE_DEVICES=0  fairseq-train     datasets/CGEC_NLPCC_2018_sample/  --save-dir  outputs/CGEC_debug_2.pinyinoff.20201214  --ddp-backend=no_c10d     --task rebert     --criterion refinement_nat_loss    --arch  levenshtein_refinement_rebert_decoder_2layers   --noise random_delete  --optimizer adam --adam-betas '(0.9,0.98)'     --lr '1e-04' --lr-scheduler inverse_sqrt     --min-lr '1e-07' --warmup-updates 10000     --warmup-init-lr '1e-06' --label-smoothing 0.1     --dropout 0.1 --weight-decay 0.01    --log-format 'simple' --log-interval 100    --save-interval-updates 10000   --max-update 2000000  --dataset-impl raw_str    --load-hf-bert-from models/bert-base-chinese/  --batch-size 6   --early-exit=2,2,2  --load-source-middle  --seed 6  --fix-bert-params  2>&1   | tee  -a   outputs/CGEC_debug_2.pinyinoff.20201214/log.log
    ```   

1. Important files:
 * fairseq/criterions/refinement_nat_loss.py # loss
 * fairseq/data/__init__.py # dataset注册相关
 * fairseq/data/indexed_dataset.py # 支持 raw_str 格式
 * fairseq/data/middle_enhanced_tokenizer_plus_language_pair_dataset.py # 数据集
 * fairseq/iterative_refinement_generator_rbt.py #infer 时候的生成器， 0.9.0 版本，暂时未跟着update 可能不work
 * fairseq/models/nat/__init__.py # model 注册相关
 * fairseq/models/nat/fairseq_nat_model.py # 一些基类
 * fairseq/models/nat/levenshtein_refinement_rebert.py # rebert model
 * fairseq/models/nat/levenshtein_utils.py # 训练的target生成
 * fairseq/models/transformer.py # 基类
 * fairseq/tasks/rebert.py # 任务类
 * fairseq/tasks/refinement.py # 任务基类



    
