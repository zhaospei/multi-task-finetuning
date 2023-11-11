import copy
import datasets

# def get_preprocessed_data(dataset_id, tokenizer, split):
#     dataset = datasets.load_dataset(dataset_id, split=split)

#     def tokenize_function(examples):
#         # print(examples)
#         model_inputs = tokenizer(['msg:\n' + vccs_msg + '\n' + diff for vccs_msg, diff in zip(examples['vccs_msg'], examples['diff'])], max_length=1024, truncation=True)
#         expl_model_inputs = tokenizer(['type:\n' + vccs_msg + '\n' + diff for vccs_msg, diff in zip(examples['vccs_msg'], examples['diff'])], max_length=1024, truncation=True)
#         model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
#         model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

#         label_output_encodings = tokenizer(examples['msg'], max_length=32, truncation=True)
#         rationale_output_encodings = tokenizer(['type_' + str(type) for type in examples['type']], max_length=32, truncation=True)

#         model_inputs['labels'] = label_output_encodings['input_ids']
#         model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

#         return model_inputs


#     tokenized_datasets = dataset.map(
#         tokenize_function,
#         remove_columns=list(dataset.features),
#         batched=True
#     )

#     return tokenized_datasets

def get_preprocessed_data(dataset_id, tokenizer, split):
    dataset = datasets.load_dataset(dataset_id, split=split)

    prompt_type = f"Give type of this code:\n{{vccs}}{{diff}}\nType:"
    prompt_msg = f"Give msg of this code:\n{{vccs}}{{diff}}\nMsg:"
    def apply_prompt_template(sample):
        return {
            "type_input": prompt_type.format(vccs=sample['vccs_msg'], diff=sample['diff']),
            "msg_input": prompt_msg.format(vccs=sample['vccs_msg'], diff=sample['diff']),
            "type_label": f"type_{{type}}".format(type=sample['type']),
            "msg_label": sample['msg']
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_function(sample):
        # print(examples)
        model_inputs = tokenizer.encode(tokenizer.bos_token + sample['msg_input'], add_special_tokens=False, max_length=991, truncation=True)
        expl_model_inputs = tokenizer.encode(tokenizer.bos_token + sample['type_input'], add_special_tokens=False, max_length=991, truncation=True)
        label_output_encodings = tokenizer.encode(sample['msg_label'] + tokenizer.eos_token, add_special_tokens=False, max_length=32, truncation=True)
        rationale_output_encodings = tokenizer.encode(sample['type_label'] + tokenizer.eos_token, add_special_tokens=False, max_length=32, truncation=True)
        # model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        # model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
        max_length = 1024 - len(model_inputs) - len(label_output_encodings)
        # mx = max(mx, len(prompt) + len(message))
        if max_length < 0:
            print("OK")

        pad_msg = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False, max_length=max_length, padding='max_length', truncation=True)
        
        max_length = 1024 - len(expl_model_inputs) - len(rationale_output_encodings)
        # mx = max(mx, len(prompt) + len(message))
        if max_length < 0:
            print("OK")

        pad_type = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False, max_length=max_length, padding='max_length', truncation=True)

        # model_inputs['labels'] = label_output_encodings['input_ids']
        # model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

        sample = {
            "input_ids": model_inputs + label_output_encodings + pad_msg,
            "attention_mask" : [1] * (len(model_inputs) + len(label_output_encodings) + len(pad_msg)),
            "labels": [-100] * len(model_inputs) + label_output_encodings + [-100] * len(pad_msg),
            "expl_input_ids": expl_model_inputs + rationale_output_encodings + pad_type,
            "expl_attention_mask": [1] * (len(expl_model_inputs) + len(rationale_output_encodings) + len(pad_type)),
            "aux_labels": [-100] * len(expl_model_inputs) + rationale_output_encodings + [-100] * len(pad_type),
        }

        return sample


    tokenized_datasets = dataset.map(
        tokenize_function,
        remove_columns=list(dataset.features),
    )

    return tokenized_datasets