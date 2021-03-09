# text_gcn

The implementation of Text GCN from the paper Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377.

The implementation of DistilBERT from the paper V. Sanh, L. Debut, J. Chaumond, and T. Wolf.   Distilbert, adistilled version of BERT: smaller, faster, cheaper and lighter.CoRR, abs/1910.01108, 2019.

## Require

Python 3.8

## Reproducing Results

1. Run `python remove_words.py 20ng`

2. Run `python build_graph.py --corpus=20ng --window_size=20 [--doc_edges]`

3. Run `python train_GCN.py [--corpus=20ng] [--num_epochs=100] [--hidden=50] [--alpha=0.05] [--verbose] [--dropout] [--print_every]`

3. Run `python train_BERT.py [--corpus=20ng] [--alpha=1e-5] [--batch_size=64] [--print_every=5] [--pretrained] [--verbose] [--num_epochs]`

4. Change `20ng` in above 3 command lines to `R8`, `R52`, `ohsumed` and `mr` when producing results for other datasets.

## Reproducing Metric Calculations

1. Run `python random_tasks.py --ttest --model_comparison=BERT`

## Reproducing t-SNE

1. Run `python random_tasks.py --t_sne`

