# Projeto Final Redes Neurais: Uma análise de propriedades mecânicas de materiais

<p>Este repositório é destinado a guardar os arquivos utilizados e desenvolvidos no projeto de consclusão da primeira parte da disciplina "Redes Neurais e Algoritmos Genéticos" do terceiro semestre do Bacharel em Ciência e Tecnologia da Ilum - Escola de Ciência do Centro Nacional de Pesquisa em Energia e Materiais (CNPEM). A questão proposta para o projeto consiste em identificar e otimizar hiperparâmetros de uma rede neural do tipo MLP (Multilayer perceptron) para resolver um problema de regressão de interesse científico.</p>
<p>Os dados utilizados para o treino e teste da rede neural desenvolvida foi encontrado no banco de dados "kaggle". O dataset tem por título "Materials and their Mechanical Properties" e trata de um conjunto de dados reais de propriedades mecânicas de diferentes materiais. Nosso projeto foi dividdo em três etapas principais, senod elas o tramento do conjunto de dados, a criação da rede neural e a otimização dos hiperparâmetros, onde cada um ficou em um notebook jupyter diferente</p>

<p>

<b>Tratamento Dataset.ipynb</b>: Iniciamos o tratamento excluindo as colunas que não seriam úteis para nossa análise. Consideramos como colunas não úteis todas que não possuissem dados númericos ou que não tivessem potencial para serem convertidos em númericos, restando apenas a coluna de identificação do material como não numerica. Após esse tratamento as seguintes colunas restaram:

<ul>
  <li>Tração máxima (Su)</li>
  <li>Limite de resistência (Sy)</li>
  <li>Módulo elástico (E)</li>
  <li>Módulo de cisalhamento (G)</li>
  <li>Coeficiente de Poisson (mu)</li>
  <li>Densidade (Ro)</li>
</ul>

Após uma análise das colunas e discussão do grupo, o atributo de Tração máxima (Su) foi escolhido como "target" para a rede neural, restando aos outros serem as "features". Após isso, as colunas foram convertidas para valores númericos e o dataset novo salvo no arquivo "Dataset.pickle"


<b>Rede Neural.ipynb</b> 
<b>Otimizando Rede.ipynb</b>

</p>
