# Integrating HuggingFace's Chinese bert base into fairseq #

--------------------------------------------------------------------------------

1. It's far from perfect but get works done.
1. Important files:
    * fairseq/data/tokenizer_dictionary.py      # dictionary that wrapping bert's tokenizer
    * fairseq/data/__init__.py                  # registers TokenizerDictionary, BertTokenizerDataset 
    * fairseq/data/data_utils.py                # adds load_indexed_raw_str_dataset() for 'raw_str' dataset
    * fairseq/data/bert_tokenizer_dataset.py    # dataset, it's dirty, and it works, for a better dataset implementation, check rebert's dataset later
    * fairseq/dataclass/constants.py            # supports raw_str dataset-impl
    * fairseq/fairseq/data/indexed_dataset.py   # supports raw_str dataset-impl
    * fairseq/models/huggingface/hf_bert.py     # the model file
    * fairseq/tasks/sentence_prediction_chinese_bert.py # task file
1. How to get your hands dirty:
    * Just run the task by
    
            $ fairseq-train datasets/cls_raw   --max-positions 510     --batch-size 6   --task sentence_prediction_chinese_bert    --required-batch-size-multiple 1   --arch hf_bert_base    --criterion sentence_prediction     --classification-head-name 'senti_head'     --num-classes 2     --dropout 0.1 --attention-dropout 0.1     --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06          --clip-norm 0.0     --lr-scheduler polynomial_decay --lr 1e-05 --total-num-update 7812  --warmup-updates 469        --max-epoch 20     --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric     --shorten-method "truncate"     --find-unused-parameters     --update-freq 1 --dataset-impl raw_str  --load-hf-bert-from models/bert-base-chinese/ --save-dir bert_ocls_raw_demo/ --skip-invalid-size-inputs-valid-test
             
    * Enjoy!
    
