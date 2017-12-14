#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np

import tensorflow as tf
import coref_model as cm
import util

import argparse, cPickle, csv, glob, itertools, shutil
#from unidecode import unidecode
from collections import Counter, defaultdict

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

import string


def create_example(text):
  raw_sentences = sent_tokenize(text)
  sentences = [word_tokenize(s) for s in raw_sentences]
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }

def update_predictions(example):
  words = util.flatten(example["sentences"])
  q_string = " ".join(words)

  for cluster in example["predicted_clusters"]:
    # first_indices = cluster[0]
    # first_phrase = " ".join(words[first_indices[0]:first_indices[1]+1])

    num_phrases = len(cluster)
    phrases = [" ".join(words[m[0]:m[1]+1]) for m in cluster]
    first_phrase = phrases[0]

    for i in range(1, num_phrases):
      if first_phrase not in phrases[i]:
        q_string = string.replace(q_string, phrases[i], first_phrase)

      # phrase = cluster[i]

      # # replace later phrases if first phrase is not a substring
      # if first_phrase not in words[phrase[0]:phrase[1]+1]:
      #   del words[phrase[0]:phrase[1]+1]
      #   words.insert(phrase[0], first_phrase)

  return q_string # " ".join(words)

def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)
  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

  example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["head_scores"] = head_scores.tolist()
  return example

def write_seq_parser_input(model, fold, out_name):
    
    reader = csv.DictReader(open(fold, 'r'), delimiter='\t')

    fieldnames = ['id', 'annotator', 'position', 'question', 'table_file', 'answer_coordinates', 'answer_text']
    mod_file = open(out_name, 'w')
    writer = csv.DictWriter(mod_file, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    prev_annotator = '0'
    pos = '0'
    qids = []
    questions = []
    tables = []
    answer_coords = []
    answers = []
    for row in reader:
        annotator = row['annotator']
        question = row['question']

        # either add question to check for coref or start anew
        if annotator != prev_annotator:    
            question_text = " ".join(questions)
            example = make_predictions(question_text, model)
            updated_qs = update_predictions(example)
            updated_qs = updated_qs.split("?")
            updated_qs = [q.strip() + "?" for q in updated_qs if q]

            # update the corefs
            for i in range(0, int(pos) + 1):
                # print(i)
                # print(len(qids))
                # print(updated_qs)
                # print(len(updated_qs))
                # print(len(tables))
                # print(len(answer_coords))
                # print(len(answers))
                # print(question)
                
                writer.writerow({'id': qids[i], 'annotator': prev_annotator, 'position': i, 'question': updated_qs[i], 'table_file': tables[i], 'answer_coordinates': answer_coords[i], 'answer_text': answers[i]})
            
            # update annotator
            prev_annotator = annotator

            # clear pos, qids, questions, tables, answer_coords, answers
            pos = '0'
            qids = []
            questions = []
            tables = []
            answer_coords = []
            answers = []

        pos = row['position']
        qids.append(row['id'])
        tables.append(row['table_file'])
        answer_coords.append(row['answer_coordinates'])
        answers.append(row['answer_text'])

        # ensure questions have '?'
        if question[-1] != '?':
          question += '?'
        questions.append(question)

    # last set
    question_text = " ".join(questions)
    example = make_predictions(question_text, model)
    updated_qs = update_predictions(example)
    updated_qs = updated_qs.split("?")
    updated_qs = [q.strip() + "?" for q in updated_qs if q]

    # update the corefs
    for i in range(0, int(pos) + 1):
      updated_q = questions[i]
      writer.writerow({'id': qids[i], 'annotator': prev_annotator, 'position': i, 'question': updated_qs[i], 'table_file': tables[i], 'answer_coordinates': answer_coords[i], 'answer_text': answers[i]})

# Use: python replace_coref.py batched input.tsv output.tsv
if __name__ == "__main__":
  util.set_gpus()

  name = sys.argv[1]
  data_path = sys.argv[2] # ../qa/SQA/data/

  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  model = cm.CorefModel(config)

  model.load_eval_data()

  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    saver.restore(session, checkpoint_path)

    write_seq_parser_input(model, data_path + 'random-split-1-train.tsv', data_path + 'mod-random-split-1-train.tsv')
    write_seq_parser_input(model, data_path + 'random-split-1-dev.tsv', data_path + 'mod-random-split-1-dev.tsv')
    write_seq_parser_input(model, data_path + 'random-split-2-train.tsv', data_path + 'mod-random-split-2-train.tsv')
    write_seq_parser_input(model, data_path + 'random-split-2-dev.tsv', data_path + 'mod-random-split-2-dev.tsv')
    write_seq_parser_input(model, data_path + 'random-split-3-train.tsv', data_path + 'mod-random-split-3-train.tsv')
    write_seq_parser_input(model, data_path + 'random-split-3-dev.tsv', data_path + 'mod-random-split-3-dev.tsv')
    write_seq_parser_input(model, data_path + 'random-split-4-train.tsv', data_path + 'mod-random-split-4-train.tsv')
    write_seq_parser_input(model, data_path + 'random-split-4-dev.tsv', data_path + 'mod-random-split-4-dev.tsv')
    write_seq_parser_input(model, data_path + 'random-split-5-train.tsv', data_path + 'mod-random-split-5-train.tsv')
    write_seq_parser_input(model, data_path + 'random-split-5-dev.tsv', data_path + 'mod-random-split-5-dev.tsv')
    write_seq_parser_input(model, data_path + 'train.tsv', data_path + 'mod-train.tsv')
    write_seq_parser_input(model, data_path + 'test.tsv', data_path + 'mod-test.tsv')
