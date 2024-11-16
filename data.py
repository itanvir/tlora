from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


def prepare_tokenized_dataloaders_mrpc(raw_datasets, model_name, batch_size=32):
    """
   Prepare tokenized dataloaders for the MRPC task.
    Verify: example_batch = next(iter(train_dataloader))
    """

    def tokenize_function(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            return_token_type_ids=True,
            truncation=True,
        )

    # Tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast_tokenizer=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2", "idx"]
    )
    tokenized_datasets.set_format("torch")

    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, val_dataloader


def prepare_tokenized_dataloaders_rte(raw_datasets, model_name, batch_size=32):
    """
    Prepare tokenized dataloaders for the RTE task.
    """

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast_tokenizer=True)

    def tokenize_function(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            return_token_type_ids=True,
            truncation=True,
        )

    # Tokenize the datasets
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2", "idx"]
    )
    tokenized_datasets.set_format("torch")

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, val_dataloader


def prepare_tokenized_dataloaders_sst2(raw_datasets, model_name, batch_size=32):
    """
    Prepare tokenized dataloaders for the SST2 task.
    """

    def tokenize_function(example):
        return tokenizer(
            example["sentence"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    # Tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast_tokenizer=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])

    # Data collator for padding (not strictly necessary here)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, val_dataloader


def prepare_tokenized_dataloaders_cola(raw_datasets, model_name, batch_size=32):
    """
    Prepare tokenized dataloaders for the COLA dataset.
    """

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast_tokenizer=True)

    def tokenize_function(example):
        return tokenizer(
            example["sentence"],
            padding=True,
            truncation=True,
            return_tensors="pt"   # Ensure this is correct for your use case
        )

    # Tokenize the datasets
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # Remove unnecessary columns
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])

    # Set the format for PyTorch
    tokenized_datasets.set_format("torch")

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, val_dataloader


def prepare_tokenized_dataloaders_qnli(raw_datasets, model_name, batch_size=32):
    """
    Prepare tokenized dataloaders for the QNLI task.
    """

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, fast_tokenizer=True)

    def tokenize_function(example):
        return tokenizer(
            example["question"],
            example["sentence"],
            return_token_type_ids=True,
            truncation=True,
        )

    # Tokenize the datasets
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["question", "sentence", "idx"]
    )
    tokenized_datasets.set_format("torch")

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, val_dataloader
