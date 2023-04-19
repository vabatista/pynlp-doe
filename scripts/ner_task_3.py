#!/usr/bin/env python
# coding=utf-8


"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset
from pynlpdoe.experiments import Dataset

import transformers
from transformers import (
	AutoConfig,
	AutoModelForTokenClassification,
	AutoTokenizer,
	DataCollatorForTokenClassification,
	HfArgumentParser,
	PretrainedConfig,
	PreTrainedTokenizerFast,
	Trainer,
	TrainingArguments,
	set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)




def main():

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	transformers.utils.logging.set_verbosity_info()

	log_level = 'INFO'
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level) 
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Detecting last checkpoint.
	last_checkpoint = None
	output_dir = '../pynlp-doe-output'
	if os.path.isdir(output_dir):
		last_checkpoint = get_last_checkpoint(output_dir)
		if last_checkpoint is not None:
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
			)

	

	dataset_name = ['conll2003']
	ds = Dataset(dataset_name)
	model_name = 'bert-base-cased'
	
	num_epochs = 1

	# Load pretrained model and tokenizer
	#
	# Distributed training:
	# The .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.
	config = AutoConfig.from_pretrained(model_name,
		num_labels=ds.num_labels,
		finetuning_task='ner'
	)

	tokenizer = AutoTokenizer.from_pretrained(model_name,
		use_fast=True
	)

	model = AutoModelForTokenClassification.from_pretrained(
		model_name,
		config=config
	)

	# Tokenizer check: this script requires a fast tokenizer.
	if not isinstance(tokenizer, PreTrainedTokenizerFast):
		raise ValueError(
			"This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
			" https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
			" this requirement"
		)

	ds.tokenizer = tokenizer # type: ignore

	# Set the correspondences label/ID inside the model config
	model.config.label2id = {l: i for i, l in enumerate(ds.label_list)}
	model.config.id2label = dict(enumerate(ds.label_list))

	# Map that sends B-Xxx label to its I-Xxx counterpart
	# b_to_i_label = []
	# for idx, label in enumerate(label_list):
	#     if label.startswith("B-") and label.replace("B-", "I-") in label_list:
	#         b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
	#     else:
	#         b_to_i_label.append(idx)

	# Preprocessing the dataset
	
	# Padding strategy
	padding = "max_length" #if data_args.pad_to_max_length else False

	train_dataset = ds.getTrainSplit()
	eval_dataset = ds.getValidationSplit()
	predict_dataset = ds.getTestSplit()
	label_list = ds.label_list
	
	# Data collator
	data_collator = DataCollatorForTokenClassification(tokenizer)

	# Metrics
	metric = evaluate.load("seqeval")

	def compute_metrics(p):
		predictions, labels = p
		predictions = np.argmax(predictions, axis=2)

		# Remove ignored index (special tokens)
		true_predictions = [
			[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]
		true_labels = [
			[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]

		results = metric.compute(predictions=true_predictions, references=true_labels)
		return {
			"precision": results["overall_precision"],
			"recall": results["overall_recall"],
			"f1": results["overall_f1"],
			"accuracy": results["overall_accuracy"],
		}

	training_args = TrainingArguments(
		output_dir=output_dir,
		evaluation_strategy='epoch',
		learning_rate=2e-5,
		#per_device_train_batch_size=16,
		#per_device_eval_batch_size=64,
		num_train_epochs=num_epochs,
		weight_decay=0.01,
		push_to_hub=False,
		logging_dir=os.path.join(output_dir,'logs'),
		#logging_steps=10,
	)

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset, 
		eval_dataset=eval_dataset, 
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics
	)

	# Training
	if True: #training_args.do_train:
		checkpoint = None
		if last_checkpoint is not None:
			checkpoint = last_checkpoint
		train_result = trainer.train(resume_from_checkpoint=checkpoint)
		metrics = train_result.metrics
		trainer.save_model()  # Saves the tokenizer too for easy upload

		metrics["train_samples"] = len(train_dataset)

		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()

	# Evaluation
	if True: # training_args.do_eval:
		logger.info("*** Evaluate ***")

		metrics = trainer.evaluate()

		
		metrics["eval_samples"] = len(eval_dataset)

		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", metrics)

	# Predict
	if True: #training_args.do_predict:
		logger.info("*** Predict ***")

		predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
		predictions = np.argmax(predictions, axis=2)

		# Remove ignored index (special tokens)
		true_predictions = [
			[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]

		trainer.log_metrics("predict", metrics)
		trainer.save_metrics("predict", metrics)

		# Save predictions
		output_predictions_file = os.path.join(output_dir, "predictions.txt")
		if trainer.is_world_process_zero():
			with open(output_predictions_file, "w") as writer:
				for prediction in true_predictions:
					writer.write(" ".join(prediction) + "\n")

	kwargs = {"finetuned_from": model_name, "tasks": "token-classification"}
	kwargs["dataset_tags"] = dataset_name

	trainer.create_model_card(**kwargs)


def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()
