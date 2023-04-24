import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
from train_cnn import train


train(embedding_type="glove", pad_pos="head", num_epochs=10, sentence_length=512, embedding_dim=300)