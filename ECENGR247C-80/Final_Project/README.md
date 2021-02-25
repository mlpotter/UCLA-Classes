# text_gcn

An PyTorch Geometric implementation of Text GCN and comparison to Logistic Regression model:

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377


## Require

See requirements.txt file

## Reproducing Results

1. Run `python remove_words.py 20ng`

2. Run `python build_graph.py --corpus=20ng`

3. Run `python train_MLP.py --corpus=20ng --verbose --num_epochs=100 --alpha=0.05 --hidden=50`

4. Change `20ng` in above 3 command lines to `R8`, `R52`, `ohsumed` and `mr` when producing results for other datasets.

## Example input data

1. `/data/20ng.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data/corpus/20ng.txt` contains raw text of each document, each line is for the corresponding line in `/data/20ng.txt`

3. `prepare_data.py` is an example for preparing your own data, note that '\n' is removed in your documents or sentences.
