from .data import Dataset
import torch

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import logging
import os, sys

import evaluate
import numpy as np

import transformers
from transformers import (
	AutoConfig,
	AutoModelForTokenClassification,
	AutoTokenizer,
	DataCollatorForTokenClassification,
	PreTrainedTokenizerFast,
	Trainer,
	TrainingArguments,
	set_seed,
)
#from transformers.trainer_utils import get_last_checkpoint


## TODO: First experiment will be evaluation of NER using 2 different models into two different training sets


logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	handlers=[logging.StreamHandler(sys.stdout)],
)




class Experiment:
	pass


class Treatment:
	dataset: Dataset
	model_name: str = ''
	task_name: str = 'ner'
	seed: int = 0
	metrics_to_collect: list = ['accuracy', 'f1', 'loss']
	output_dir = '../pynlpdoe-output'
	label_list: list = []
	metric = None
 
	def __init__(self, dataset: Dataset, model_name, seed):
		self.model_name = model_name
		self.dataset = dataset
		self.seed = seed
  
  
	def compute_metrics(self, p):
		predictions, labels = p
		predictions = np.argmax(predictions, axis=2)
		label_list = self.label_list
		metric = self.metric
		# Remove ignored index (special tokens)
		true_predictions = [
			[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]
		true_labels = [
			[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]

		results = self.metric.compute(predictions=true_predictions, references=true_labels)  # type: ignore
		return {
			"precision": results["overall_precision"], # type: ignore
			"recall": results["overall_recall"], # type: ignore
			"f1": results["overall_f1"], # type: ignore
			"accuracy": results["overall_accuracy"], # type: ignore
		}
  
	def exec_treatment(self):

		transformers.utils.logging.set_verbosity_info()

		log_level = 'INFO'
		logger.setLevel(log_level)
		transformers.utils.logging.set_verbosity(0)
		transformers.utils.logging.enable_default_handler()
		transformers.utils.logging.enable_explicit_format()

		set_seed(self.seed)
		torch.manual_seed(self.seed)
		np.random.seed(self.seed)


		ds = self.dataset
		model_name = self.model_name
		
		num_epochs = 1 #TODO: this is a Factor into the treatment

		# Load pretrained model and tokenizer
		config = AutoConfig.from_pretrained(model_name,
			num_labels=ds.num_labels,
			finetuning_task=self.task_name
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
		train_dataset = ds.getTrainSplit()
		eval_dataset = ds.getValidationSplit()
		predict_dataset = ds.getTestSplit()

		# Set the correspondences label/ID inside the model config
		model.config.label2id = {l: i for i, l in enumerate(ds.label_list)}
		model.config.id2label = dict(enumerate(ds.label_list))

		label_list = ds.label_list
		# Data collator
		data_collator = DataCollatorForTokenClassification(tokenizer)

		# Metrics
		self.metric = evaluate.load("seqeval")

		training_args = TrainingArguments(
			output_dir=self.output_dir,
			evaluation_strategy='epoch',
			learning_rate=2e-5,
			num_train_epochs=num_epochs,
			weight_decay=0.01,
			push_to_hub=False,
			logging_dir=os.path.join(self.output_dir,'logs'),
			#logging_steps=10,
		)

		# Initialize our Trainer
		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset, # type: ignore
			eval_dataset=eval_dataset, # type: ignore
			tokenizer=tokenizer,
			data_collator=data_collator,
			compute_metrics=self.compute_metrics # type: ignore
   			#self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids),  self.metrics_extra_args)
		)

		# Training
		checkpoint = None
		# if last_checkpoint is not None:
		# 	checkpoint = last_checkpoint
		train_result = trainer.train(resume_from_checkpoint=checkpoint)
		metrics = train_result.metrics
		trainer.save_model()  # Saves the tokenizer too for easy upload

		metrics["train_samples"] = len(train_dataset)

		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()

		# Predict
		logger.info("*** Predict ***")

		predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict") # type: ignore
		predictions = np.argmax(predictions, axis=2)

		# Remove ignored index (special tokens)
		true_predictions = [
			[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels) # type: ignore
		]

		trainer.log_metrics("predict", metrics)
		trainer.save_metrics("predict", metrics)

		# Save predictions
		output_predictions_file = os.path.join(self.output_dir, "predictions.txt")
		if trainer.is_world_process_zero():
			with open(output_predictions_file, "w") as writer:
				for prediction in true_predictions:
					writer.write(" ".join(prediction) + "\n")

		return metrics
