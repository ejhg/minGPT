import unittest

from mingpt.bpe import get_encoder


class TestBPE(unittest.TestCase):

    def test(self):
        text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
        e = get_encoder()
        r = e.encode(text)
        self.assertEquals(r,
                          [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83,
                           1058, 35, 12520, 97, 245])


if __name__ == '__main__':
    unittest.main()
