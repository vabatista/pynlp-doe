from transformers import AutoTokenizer, pipeline
from .data import Dataset
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
## https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
## TODO: First experiment will be evaluation of NER using 2 different models into two different training sets


logger = logging.getLogger(__name__)

class Experiment:
	pass


class Treatment:
	train_ds = None
	infer_ds = None
	model_name = None
	task_name = None
	def __init__(self, train_ds: Dataset, infer_ds: Dataset, model_name, task_name):
		self.train_ds = train_ds
		self.infer_ds = infer_ds
		self.model_name = model_name
		self.task_name = task_name

	def exec_inference(self, ds: Dataset):
		#pipe = pipeline(self.task_name, self.model_name)
		#return pipe(ds)
		tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		model = AutoModelForTokenClassification.from_pretrained(
			self.model_name,
			use_auth_token=True
		)  
  
        predictions, labels, metrics = model.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

class Training:
	pass


