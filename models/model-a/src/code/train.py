import argparse
import logging
import time
import pandas as pd
import torch
from model import TextClassificationModel
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.datasets import DATASETS, AG_NEWS
from torchtext.prototype.transforms import load_sp_model, PRETRAINED_SP_MODEL, SentencePieceTokenizer
from torchtext.utils import download_from_url
from torchtext.vocab import build_vocab_from_iterator
import boto3
import os
import tarfile

s3 = boto3.client('s3')

r"""
This file shows the training process of the text classification model.
"""


def yield_tokens(data_iter, ngrams):
    for _, text in data_iter:
        yield ngrams_iterator(tokenizer(text), ngrams)


def save_to_s3(local_path, bucket_name, s3_key):
    """
    Save a local file to an S3 bucket.

    :param local_path: Path to the local file.
    :param bucket_name: Name of the S3 bucket.
    :param s3_key: S3 key (path in bucket).
    """
    if os.path.exists(local_path):
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    else:
        print(f"File {local_path} does not exist.")


def split_and_save_data_to_s3(dataset, bucket_name, train_prefix, test_prefix):
    """
    Splits the dataset into training and testing sets and uploads them to S3.

    :param dataset: The dataset to be split.
    :param bucket_name: Name of the S3 bucket.
    :param train_prefix: Prefix for training data in S3.
    :param test_prefix: Prefix for testing data in S3.
    """
    # Split dataset into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    # Convert datasets to Pandas DataFrames
    train_data = [(label, text) for label, text in train_dataset]
    test_data = [(label, text) for label, text in test_dataset]

    train_df = pd.DataFrame(train_data, columns=['label', 'text'])
    test_df = pd.DataFrame(test_data, columns=['label', 'text'])

    # Save locally as CSV files
    train_local_path = '/tmp/train.csv'
    test_local_path = '/tmp/test.csv'

    print("Saving training and testing datasets locally...")
    train_df.to_csv(train_local_path, index=False)
    test_df.to_csv(test_local_path, index=False)

    # Upload CSV files to S3
    print("Uploading training and testing datasets to S3...")
    save_to_s3(train_local_path, bucket_name, train_prefix)
    save_to_s3(test_local_path, bucket_name, test_prefix)


def create_tar_file(output_filename, files_to_add):
    """
    Create a tar.gz file containing the specified files and directories.

    :param output_filename: Name of the output tar.gz file.
    :param files_to_add: List of file paths to include in the tar.gz file.
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        for file_path in files_to_add:
            if os.path.exists(file_path):
                tar.add(file_path, arcname=os.path.basename(file_path))
                print(f"Added {file_path} to {output_filename}")
            else:
                print(f"File or directory {file_path} does not exist.")


def save_tar_to_s3(local_tar_path, bucket_name, s3_key):
    """
    Upload the tar.gz file to an S3 bucket.

    :param local_tar_path: Path to the local tar.gz file.
    :param bucket_name: Name of the S3 bucket.
    :param s3_key: S3 key (path in bucket).
    """
    if os.path.exists(local_tar_path):
        s3.upload_file(local_tar_path, bucket_name, s3_key)
        print(f"Uploaded {local_tar_path} to s3://{bucket_name}/{s3_key}")
    else:
        print(f"File {local_tar_path} does not exist.")


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def train(dataloader, model, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(epoch, idx, len(
                    dataloader), total_acc / total_count)
            )
            total_acc, total_count = 0, 0


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a text classification model on text classification datasets.")
    parser.add_argument("--dataset", type=str, default="AG_NEWS",
                        help="Dataset to use for training (default: AG_NEWS)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="num epochs (default=5)")
    parser.add_argument("--embed-dim", type=int, default=32,
                        help="embed dim. (default=32)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="batch size (default=16)")
    parser.add_argument("--split-ratio", type=float, default=0.95,
                        help="train/valid split ratio (default=0.95)")
    parser.add_argument("--learning-rate", type=float, default=4.0,
                        help="learning rate (default=4.0)")
    parser.add_argument("--lr-gamma", type=float, default=0.8,
                        help="gamma value for lr (default=0.8)")
    parser.add_argument("--ngrams", type=int, default=2,
                        help="ngrams (default=2)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="num of workers (default=1)")
    parser.add_argument("--device", default="cpu", help="device (default=cpu)")
    parser.add_argument("--data-dir", default=".data",
                        help="data directory (default=.data)")
    parser.add_argument(
        "--use-sp-tokenizer", type=bool, default=False, help="use sentencepiece tokenizer (default=False)"
    )
    parser.add_argument("--dictionary", help="path to save vocab")
    parser.add_argument(
        "--save-model-path", default="/opt/ml/model/model.pth", help="path for saving model")
    parser.add_argument("--dictionary_path", default="/opt/ml/model/vocab.pth")
    parser.add_argument("--logging-level", default="WARNING",
                        help="logging level (default=WARNING)")
    # S3 related arguments
    parser.add_argument("--bucket-name", default=os.getenv('BUCKET_NAME', 'textclassificationmldemo-model-archiving'),
                        help="Name of the S3 bucket where data and artifacts will be stored.")

    parser.add_argument("--train-prefix", default="models/model-a/training/data/train.csv",
                        help="S3 prefix for training data.")

    parser.add_argument("--test-prefix", default="models/model-a/training/data/test.csv",
                        help="S3 prefix for testing data.")

    parser.add_argument("--artifacts-prefix", default="models/model-a/train-artifacts",
                        help="S3 prefix for saving model artifacts.")

    # Directories for input/output data and model artifacts
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str,
                        default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load dataset
    print("Loading AG_NEWS dataset...")

    dataset = list(AG_NEWS(split='train'))

    # Split data and upload to S3
    split_and_save_data_to_s3(dataset,
                              args.bucket_name,
                              args.train_prefix,
                              args.test_prefix)

    num_epochs = args.epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.learning_rate
    device = args.device
    data_dir = args.data_dir
    split_ratio = args.split_ratio
    ngrams = args.ngrams
    use_sp_tokenizer = args.use_sp_tokenizer

    logging.basicConfig(level=getattr(logging, args.logging_level))

    if use_sp_tokenizer:
        sp_model_path = download_from_url(
            PRETRAINED_SP_MODEL["text_unigram_15000"], root=data_dir)
        sp_model = load_sp_model(sp_model_path)
        tokenizer = SentencePieceTokenizer(sp_model)
    else:
        tokenizer = get_tokenizer("basic_english")

    train_iter = DATASETS[args.dataset](root=data_dir, split="train")
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, ngrams), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x):
        return vocab(list(ngrams_iterator(tokenizer(x), ngrams)))

    def label_pipeline(x):
        return int(x) - 1

    train_iter = DATASETS[args.dataset](root=data_dir, split="train")
    num_class = len(set([label for (label, _) in train_iter]))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    model = TextClassificationModel(
        len(vocab), embed_dim, num_class).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    train_iter, test_iter = DATASETS[args.dataset]()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(
        split_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(
        split_valid_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        train(train_dataloader, model, optimizer, criterion, epoch)
        accu_val = evaluate(valid_dataloader, model)
        scheduler.step()
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val)
        )
        print("-" * 59)

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader, model)
    print("test accuracy {:8.3f}".format(accu_test))

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to("cpu"), args.save_model_path)

    if args.dictionary is not None:
        print("Save vocab to {}".format(args.dictionary))
        torch.save(vocab, args.dictionary)

    # Save model and vocab to output directory
    torch.save(model.state_dict(), args.save_model_path)
    torch.save(vocab, args.dictionary_path)

   # Paths for model artifacts and code folder
    model_path = args.save_model_path  # Example: "/opt/ml/model/model.pth"
    vocab_path = args.dictionary_path  # Example: "/opt/ml/model/vocab.pth"
    code_folder = os.path.join(os.getcwd(), "models/model-a/src/code")

    # Ensure all paths exist
    if not os.path.exists(code_folder):
        print(f"Code folder does not exist: {code_folder}")

    # Tar file output path
    tar_output_path = "/tmp/model.tar.gz"

    # Create a tar.gz file containing model.pth, vocab.pth, and the code folder
    create_tar_file(
        tar_output_path,
        [model_path, vocab_path, code_folder]
    )

    # Upload the tar.gz file to S3
    bucket_name = args.bucket_name
    # Example: "models/model-a/train-artifacts/model.tar.gz"
    s3_key = f"{args.artifacts_prefix}/model.tar.gz"

    save_tar_to_s3(tar_output_path, bucket_name, s3_key)
