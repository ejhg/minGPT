"""
Trains a character-level language model.
"""

import unittest

import os
import torch
from torch.utils.data import Dataset

from mingpt.model.gpt import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode

input_file = os.path.join(os.path.dirname(__file__), 'tinyshakespeare.txt')


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class TestAdder(unittest.TestCase):

    def test(self):
        config = CfgNode()
        config.system = CfgNode()

        config.model = GPT.get_default_config()
        config.model.n_layer = 6
        config.model.n_head = 6
        config.model.n_embd = 192

        print(config)

        text = open(input_file, 'r').read()

        data_config = CfgNode()
        data_config.block_size = 64
        train_dataset = CharDataset(data_config, text)

        config.model.vocab_size = train_dataset.get_vocab_size()
        config.model.block_size = train_dataset.get_block_size()
        model = GPT(config.model)

        trainer_config = Trainer.get_default_config()
        trainer_config.max_iters = 1000
        trainer_config.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster
        trainer = Trainer(trainer_config, model, train_dataset)

        def batch_end_callback(trainer):
            if trainer.iter_num % 10 == 0:
                print(
                    f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

            if trainer.iter_num % 100 == 0:
                # evaluate both the train and tests score
                model.eval()
                with torch.no_grad():
                    # sample from the model...
                    context = "The"
                    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(
                        trainer.device)
                    y = model.generate(x, data_config.block_size, temperature=1.0, do_sample=True, top_k=10)[0]
                    completion = ''.join([train_dataset.itos[int(i)] for i in y])
                    print(completion)
                # revert model to training mode
                model.train()

        trainer.set_callback('on_batch_end', batch_end_callback)

        # run the optimization
        trainer.run()

        self.assertLessEqual(trainer.loss.item(), 1.8)


if __name__ == '__main__':
    unittest.main()
