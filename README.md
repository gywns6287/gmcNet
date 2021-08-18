# gmcNet: gene module clustering Network in WGCNA

## Model summary

To identify desired gene module in WGCNA, we proposed the gmcNet. gmcNet is a GNN-based clsutering algorithm, which can cluster genes according to the co-expression topology (genes in the same module should be strongly connected) and to the single-level expression (genes in the same module should have similar expression patterns). The key innovation of gmcNet is incorporating the single-expression of genes with co-expression of their neighbor genes.

### Model Input
gmcNet requries four inputs to implement unsupervised clustering. Let, <img alt="$n$" src="svgs/c068b57af6b6fa949824f73dcb828783.png?invert_in_darkmode" align=middle width="42.05817pt" height="22.407pt"/>. the is number of genes and $m$ is the number of expression sample.
1. $\textbf{X}\in\mathbb{R}^{n \times m}$ : Single-expression features of $n$ genes.
2. $\textbf{T}\in\mathbb{R}^{n \times n}$ : Topological overlap matrix, which is created using the topological overlap measure between $n$ genes.
3. $\textbf{T}_\textbf{p}\in\mathbb{R}^{n \times n}$ :  Topological overlap matrix, which is created only with gene pairs of positive correlation coefficient.
4. $\textbf{T}_\textbf{n}\in\mathbb{R}^{n \times n}$ :  Topological overlap matrix, which is created only with gene pairs of neagtive correlation coefficient.

### Network structure
gmcNet includes a co-expression pattern recognizer (CEPR) and module classifier. 
![fig8](https://user-images.githubusercontent.com/71325306/129822771-2f515fd4-00db-4de7-8b24-936298c1ca00.png)
**CEPR** : With massage passing operation, CEPR generates the embedding feature $\bar{\textbf{X}}\in\mathbb{R}^{n \times m'}$, which accounts for single-epxression and two diffrent co-expressions in $m'$ dimension. 

**Module classifier** : Given CEPR-embedding feature $\bar{\textbf{X}}$, the module classifier computes module-assignment probability  $\textbf{M}\in\mathbb{R}^{n \times k}$ using a multi-layer perceptron (MLP), where $k$ is the number of modules. Finally, $i$th-row of  $\textbf{M}$ corresponds to module assifnment probability of gene $i$. In other words, gene $i$ belongs to module $c$ if $\textbf{M}_{ic}$ is the maximum value of the $i$th-row of $\textbf{M}$.

## Implementation
