# GRNN-SR

Requirements
- Python 3.5+
- Tensorflow 0.12.1

Main files
- reader.py (reader for loading and preprocessing data)
- base.py   (base model)
- cell.py   (cells of GRNN-SR and GRNN-SP)
- model.py  (training and testing of different models)
- data      (directory of dataset)

Outputs
- checkpoints (saving of the model)
- results (results of test)
    - results.txt
    - negation_results.txt
    - negation_results_probs.txt
    - intensity_results.txt
    - intensity_results_probs.txt

Usage
- Run "$ python model.py" in command line or
- Open the project in PyCharm and run the "model.py" file

Optional Arguments:
  -h, --help            show this help message and exit
  --unrolled_lstm [UNROLLED_LSTM]
                        use a statically unrolled LSTM instead of dynamic_rnn
  --nounrolled_lstm
  --learning_rate LEARNING_RATE
                        Learning rate of Adam optimizer (default: 0.001)
  --hidden_dim HIDDEN_DIM
                        The dimension of hidden layer (default: 300)
  --embed_dim EMBED_DIM
                        The dimentsion of word embeddings (default: 300)
  --batch_size BATCH_SIZE
                        Batch size (default: 32)
  --epochs EPOCHS       Number of training epochs (default: 2)
  --dataset DATASET     The name of dataset from [SST, movie] (default: SST)
  --encoder_type ENCODER_TYPE
                        The type of encoder from [GRU, LSTM, BiLSTM, GRNNSR,
                        GRNNSP] (defalut: GRU)
  --checkpoint_dir CHECKPOINT_DIR
                        Directory name to save the checkpoints (default:
                        checkpoints)
  --binary [BINARY]     True for binary classification and False for 5-class
                        classification (default: True)
