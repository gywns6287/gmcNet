# gmcNet: gene module clustering Network in WGCNA

## Model summary

To identify desired gene module in WGCNA, we proposed the gmcNet. gmcNet is a GNN-based clsutering algorithm, which can cluster genes according to the co-expression topology (genes in the same module should be strongly connected) and to the single-level expression (genes in the same module should have similar expression patterns). The key innovation of gmcNet is incorporating the single-expression of genes with co-expression of their neighbor genes.

### Model Input
gmcNet requries four inputs to implement unsupervised clustering. Let, <img src="https://render.githubusercontent.com/render/math?math=n"> is the number of genes and <img src="https://render.githubusercontent.com/render/math?math=m"> is the number of expression sample.
1. <img src="https://render.githubusercontent.com/render/math?math=\textbf{X}\in\mathbb{R}^{n \times m}"> : Single-expression features of <img src="https://render.githubusercontent.com/render/math?math=n"> genes.
2. <img src="https://render.githubusercontent.com/render/math?math=\textbf{T}\in\mathbb{R}^{n \times n}"> : Topological overlap matrix, which is created using the topological overlap measure between <img src="https://render.githubusercontent.com/render/math?math=n"> genes.
3. <img src="https://render.githubusercontent.com/render/math?math=\textbf{T}_\textbf{p}\in\mathbb{R}^{n \times n}"> :  Topological overlap matrix, which is created only with gene pairs of positive correlation coefficient.
4. <img src="https://render.githubusercontent.com/render/math?math=\textbf{T}_\textbf{n}\in\mathbb{R}^{n \times n}"> :  Topological overlap matrix, which is created only with gene pairs of neagtive correlation coefficient.

### Network structure
gmcNet includes a co-expression pattern recognizer (CEPR) and module classifier. 

![fig8](https://user-images.githubusercontent.com/71325306/129822771-2f515fd4-00db-4de7-8b24-936298c1ca00.png)

**CEPR** : With massage passing operation, CEPR generates the embedding feature <img src="https://render.githubusercontent.com/render/math?math=\bar{\textbf{X}}\in\mathbb{R}^{n \times m'}">, which accounts for single-epxression and two diffrent co-expressions in <img src="https://render.githubusercontent.com/render/math?math=m'"> dimension. 

**Module classifier** : Given CEPR-embedding feature <img src="https://render.githubusercontent.com/render/math?math=\bar{\textbf{X}}">, the module classifier computes module-assignment probability  <img src="https://render.githubusercontent.com/render/math?math=\textbf{M}\in\mathbb{R}^{n \times k}"> using a multi-layer perceptron (MLP), where <img src="https://render.githubusercontent.com/render/math?math=k"> is the number of modules. Finally, <img src="https://render.githubusercontent.com/render/math?math=i">th-row of  <img src="https://render.githubusercontent.com/render/math?math=\textbf{M}"> corresponds to module assifnment probability of gene <img src="https://render.githubusercontent.com/render/math?math=i">. In other words, gene <img src="https://render.githubusercontent.com/render/math?math=i"> belongs to module <img src="https://render.githubusercontent.com/render/math?math=c"> if <img src="https://render.githubusercontent.com/render/math?math=\textbf{M}_{ic}"> is the maximum value of the <img src="https://render.githubusercontent.com/render/math?math=i">th-row of <img src="https://render.githubusercontent.com/render/math?math=\textbf{M}">.

## Implementation

### 1. Preparing
our models were implemented by **tensorflow 2.3** in **Python 3.8.6**

#### 1.1. Requirements
  
Requirements  can be installed through the following command in your shell.
```
pip install -r [CODE PATH]/requirements.txt
```
#### 1.2. Input Data

**expr**: gene expression data. A text file with a header line, and then one line per sample with  <img src="https://render.githubusercontent.com/render/math?math=n+1"> columns. the first column is gene name.
