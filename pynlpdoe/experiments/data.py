# type: ignore
from transformers import AutoTokenizer
from datasets import ClassLabel, load_dataset
from datasets import DatasetDict

class Dataset:
	hf_dataset: DatasetDict
	column_names: list = []
	feature: list = []
	num_labels: int = 0
	label_to_id: dict = {}
	label_list = []
	tokenizer: AutoTokenizer
	NUM_PROC = 4
	label_column_name: str = "ner_tags"     

	## Receive parameters to load hugging face dataset. 
	## For instance, dataset tner/wikineural has several subsets for each language and
	## needs to pass the language also: ['tner/wikineural', 'pt']
	## Another exampe: ['wnut_17']
	def __init__(self, dataset_params: list):
		#TODO: validate parameters
		self.hf_dataset = load_dataset(*dataset_params)		
		if "train" not in self.hf_dataset:
			raise ValueError("--do_train requires a train dataset")
		if 'ner_tags' in self.hf_dataset['train'].features:
			self.label_column_name = "ner_tags"
		elif 'tags' in self.hf_dataset['train'].features:
			self.label_column_name = "tags"	
		else:
			self.label_column_name = ""	
   
		self.column_names = self.hf_dataset["train"].column_names
		self.features = self.hf_dataset["train"].features 
		self._pre_process_dataset()
		pass

	def getTrainSplit(self):
		return self.hf_dataset['train'].map(
			self._tokenize_and_align_labels,
			batched=True,
			num_proc=self.NUM_PROC,
			desc="Tokenizing train dataset",
		)
	
	def getValidationSplit(self):
		return self.hf_dataset['validation'].map(
			self._tokenize_and_align_labels,
			batched=True,
			num_proc=self.NUM_PROC,
			desc="Tokenizing validation dataset",
		)
	
	def getTestSplit(self):
		return self.hf_dataset['test'].map(
			self._tokenize_and_align_labels,
			batched=True,
			num_proc=self.NUM_PROC,
			desc="Tokenizing test dataset",
		)

	def _pre_process_dataset(self):
		
		labels_are_int = isinstance(self.features[self.label_column_name].feature, ClassLabel)
		if labels_are_int:
			label_list = self.features[self.label_column_name].feature.names
			self.label_to_id = {i: i for i in range(len(label_list))}
		else:
			label_list = self._get_label_list(self.hf_dataset["train"][self.label_column_name])
			self.label_to_id = {l: i for i, l in enumerate(label_list)}
		self.label_list = label_list
		self.num_labels = len(label_list)     
	
	def _get_label_list(self, labels):
		unique_labels = set()
		for label in labels:
			unique_labels = unique_labels | set(label)
		label_list = list(unique_labels)
		label_list.sort()
		return label_list


	# Tokenize all texts and align the labels with them.
	def _tokenize_and_align_labels(self, examples):
		# Padding strategy
		padding = "max_length"
		text_column_name = "tokens"
		tokenizer = self.tokenizer
		tokenized_inputs = tokenizer(
			examples[text_column_name],
			padding=padding,
			truncation=True,
			max_length=None,
			# For NER we use this argument because the texts in our dataset are lists of words (with a label for each word).
			is_split_into_words=True,
		)
		labels = []
		for i, label in enumerate(examples[self.label_column_name]):
			word_ids = tokenized_inputs.word_ids(batch_index=i)
			previous_word_idx = None
			label_ids = []
			for word_idx in word_ids:
				# Special tokens have a word id that is None. We set the label to -100 so they are automatically
				# ignored in the loss function.
				if word_idx is None:
					label_ids.append(-100)
				# We set the label for the first token of each word.
				elif word_idx != previous_word_idx:
					label_ids.append(self.label_to_id[label[word_idx]])
				# For the other tokens in a word, we set the label to -100
				else:
					label_ids.append(-100)
				previous_word_idx = word_idx

			labels.append(label_ids)
		tokenized_inputs["labels"] = labels
		return tokenized_inputs
	

