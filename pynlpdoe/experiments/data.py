from datasets import ClassLabel, load_dataset
from datasets import DatasetDict

class Dataset:
	hf_dataset = None

	## Receive parameters to load hugging face dataset. 
	## For instance, dataset tner/wikineural has several subsets for each language and
	## needs to pass the language also: ['tner/wikineural', 'pt']
	## Another exampe: ['wnut_17']
	def __init__(self, dataset_params: list) -> DatasetDict:
		self.hf_dataset = load_dataset(*dataset_params)
		
	
	def getTrainSplit(self):
		return hf_dataset['train']
	
	def getValidationSplit(self):
		return hf_dataset['validation']
	
	def getTestSplit(self):
		return hf_dataset['test']

	
	def tokenize_and_align_labels(self, examples, tokenizer):
		tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

		labels = []
		for i, label in enumerate(examples[f"ner_tags"]):
			word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
			previous_word_idx = None
			label_ids = []
			for word_idx in word_ids:  # Set the special tokens to -100.
				if word_idx is None:
					label_ids.append(-100)
				elif word_idx != previous_word_idx:  # Only label the first token of a given word.
					label_ids.append(label[word_idx])
				else:
					label_ids.append(-100)
				previous_word_idx = word_idx
			labels.append(label_ids)

		tokenized_inputs["labels"] = labels
		return tokenized_inputs