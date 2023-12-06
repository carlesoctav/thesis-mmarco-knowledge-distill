from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

import os
import logging
import gzip
import numpy as np
import sys
import zipfile
import io
from shutil import copyfile
import csv
import sys
import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    teacher_model_name = 'sentence-transformers/msmarco-bert-base-dot-v5'  
    student_model_name = 'bert-base-multilingual-cased'  

    max_seq_length = 256  
    train_batch_size = 32  
    inference_batch_size = 32  
    max_sentences_per_language = 500_000  
    train_max_sentence_length = 10000 

    num_epochs = 5  
    num_warmup_steps = 0.1* (500_000)/32 *5

    num_evaluation_steps = 50000  # Evaluate performance after every xxxx steps

    output_path = "output/make-multilingual-large-msmarco-" + teacher_model_name.replace("/", "_") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Write self to path
    os.makedirs(output_path, exist_ok=True)

    train_script_path = os.path.join(output_path, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    train_files = []
    dev_files = []
    is_dev_file = False
    for arg in sys.argv[1:]:
        if arg.lower() == '--dev':
            is_dev_file = True
        else:
            if not os.path.exists(arg):
                print("File could not be found:", arg)
                exit()

            if is_dev_file:
                dev_files.append(arg)
            else:
                train_files.append(arg)

    if len(train_files) == 0:
        print("Please pass at least some train files")
        print("python make_multilingual_sys.py file1.tsv.gz file2.tsv.gz --dev dev1.tsv.gz dev2.tsv.gz")
        exit()

    logging.info("Train files: {}".format(", ".join(train_files)))
    logging.info("Dev files: {}".format(", ".join(dev_files)))

    
    logging.info("Load teacher model")
    teacher_model = SentenceTransformer(teacher_model_name)
    teacher_model.to('cuda')
    teacher_model.max_seq_length = max_seq_length

    logging.info("Create student model from scratch")
    word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls")
    student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=False)
    for train_file in train_files:
        train_data.load_data(train_file, max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MSELoss(model=student_model)

    #### Evaluate cross-lingual performance on different tasks #####
    mse_evaluators = []  # evaluators has a list of different evaluator classes we call periodically
    trans_evaluator = []

    for dev_file in dev_files:
        logging.info("Create evaluator for " + dev_file)
        src_sentences = []
        trg_sentences = []
        with gzip.open(dev_file, 'rt', encoding='utf8') if dev_file.endswith('.gz') else open(dev_file, encoding='utf8') as fIn:
            for line in fIn:
                splits = line.strip().split('\t')
                if splits[0] != "" and splits[1] != "":
                    src_sentences.append(splits[0])
                    trg_sentences.append(splits[1])

        # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
        dev_mse = evaluation.MSEEvaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file), teacher_model=teacher_model, batch_size=inference_batch_size)
        mse_evaluators.append(dev_mse)

        dev_trans_acc = evaluation.TranslationEvaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file), batch_size=inference_batch_size)
        trans_evaluator.append(dev_trans_acc)

    

   
    student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluation.SequentialEvaluator(mse_evaluators + trans_evaluator, main_score_function=lambda scores: np.mean(scores[0:len(mse_evaluators)])),
                      epochs=num_epochs,
                      warmup_steps=num_warmup_steps,
                      evaluation_steps=num_evaluation_steps,
                      output_path=output_path,
                      save_best_model=True,
                      optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}, use_amp=True, output_path_ignore_not_empty=True
                      )


# Script was called via:
#python make_multilingual_msmarco_sys.py parallel-sentences/msmarco/msmarco-queries.train.en-de.tsv parallel-sentences/msmarco/msmarco-corpus.train.en-de.tsv parallel-sentences/TED2020/TED2020-en-de-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-de-train.tsv.gz --dev parallel-sentences/msmarco/msmarco-queries.dev.en-de.tsv parallel-sentences/msmarco/msmarco-corpus.dev.en-de.tsv parallel-sentences/TED2020/TED2020-en-de-dev.tsv.gz