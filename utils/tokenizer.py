import copy
import datasets

def get_preprocessed_data(dataset_id, tokenizer, split):
    dataset = datasets.load_dataset(dataset_id, split=split)

    def tokenize_function(examples):
        model_inputs = tokenizer('msg: ' + examples['vccs_msg'] + examples['diff'], max_length=1024, truncation=True)
        expl_model_inputs = tokenizer('type: ' + examples['vccs_msg'] + examples['diff'], max_length=1024, truncation=True)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

        label_output_encodings = tokenizer(examples['msg'], max_length=32, truncation=True)
        rationale_output_encodings = tokenizer(examples['type'], max_length=32, truncation=True)

        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

        return model_inputs


    tokenized_datasets = dataset.map(
        tokenize_function,
        remove_columns=['input', 'rationale', 'label', 'llm_label'],
        batched=True
    )

    return tokenized_datasets