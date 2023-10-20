from datasets import load_dataset, concatenate_datasets

dataset = load_dataset("carlesoctav/en-id-parallel-sentences")

dataset_1 = dataset["msmarcoquery"]

for split in dataset:
    if split == "msmarcocollection":
        dataset_1 = concatenate_datasets([dataset_1, dataset[split]])

dataset_1.push_to_hub("carles-undergrad-thesis/en-id-parallel-sentences")
