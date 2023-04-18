from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
import datasets

def fine_tune_model(model_name, num_epochs, dataset_name):
    # Load the pre-trained model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the dataset
    dataset = datasets.load_dataset(dataset_name)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Define the data collator
    def data_collator(features):
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            input_ids.append(feature['input_ids'])
            attention_mask.append(feature['attention_mask'])
            labels.append(feature['ner_tags'])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=datasets.load_metric('seqeval'),
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model on the test set
    test_results = trainer.evaluate(dataset['test'])
    print(test_results)

    
fine_tune_model('bert-base-cased', 1, 'conll2003')