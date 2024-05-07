# Projeto Final Redes Neurais: Uma análise de propriedades mecânicas de materiais

<p align = 'justify'>Este repositório é destinado a guardar os arquivos utilizados e desenvolvidos no projeto de consclusão da primeira parte da disciplina "Redes Neurais e Algoritmos Genéticos" do terceiro semestre do Bacharel em Ciência e Tecnologia da Ilum - Escola de Ciência do Centro Nacional de Pesquisa em Energia e Materiais (CNPEM). A questão proposta para o projeto consiste em identificar e otimizar hiperparâmetros de uma rede neural do tipo MLP (Multilayer perceptron) para resolver um problema de regressão de interesse científico.</p>
<p align = 'justify'>Os dados utilizados para o treino e teste da rede neural desenvolvida foi encontrado no banco de dados "kaggle". O dataset tem por título "Materials and their Mechanical Properties" e trata de um conjunto de dados reais de propriedades mecânicas de diferentes materiais. Nosso projeto foi dividdo em três etapas principais, senod elas o tramento do conjunto de dados, a criação da rede neural e a otimização dos hiperparâmetros, onde cada um ficou em um notebook jupyter diferente. O leitor deve se sentir a vontade quanto à ordem de execução dos notebooks abaixo, porém, é altamente recomendado que o mesmo siga o fluxo descrito abaixo para um melhor entendimento sobre o processo de criação da equipe. </p>



<p align = 'justify'><b>Tratamento Dataset.ipynb</b>: Iniciamos o tratamento excluindo as colunas que não seriam úteis para nossa análise. Consideramos como colunas não úteis todas que não possuissem dados númericos ou que não tivessem potencial para serem convertidos em númericos, restando apenas a coluna de identificação do material como não numerica. Após esse tratamento as seguintes colunas restaram:</p>

<ul>
  <li>Tração máxima (Su);</li>
  <li>Limite de resistência (Sy);</li>
  <li>Módulo elástico (E);</li>
  <li>Módulo de cisalhamento (G);</li>
  <li>Coeficiente de Poisson (mu);</li>
  <li>Densidade (Ro).</li>
</ul>

<p align = 'justify'>Após uma análise das colunas e discussão do grupo, o atributo de Tração máxima (Su) foi escolhido como "target" para a rede neural, restando aos outros serem as "features". Após isso, as colunas foram convertidas para valores númericos e o dataset novo salvo no arquivo "Dataset.pickle"</p>


<p align = 'justify'><b>Rede Neural.ipynb</b>: Dividimos o dataset dos dados escolhidos já tratados em teste e validação juntamente com uma ferramenta apropriada para encapsular esse pré-processamento. Uma rede neural do tipo MLP (Multilauyer perceptron) foi criada e treinada utilizando o treinador do pytorch lightning ajustando a rede criada com os dados encapsulados. Após isso foram feitas análises da curva de aprendizado da rede com o treino e validação e uma comparação dos valores reais do conjunto de dados com os preditos pela rede</p>


<p align = 'justify'><b>Otimizando Rede.ipynb</b> Fizemos a otimização dos hiperparâmetros da rede neural utilizando o optuna. Para isso, desenvolvemos uma função para criação do modelo que em si já cria uma rede neural com os dados sugeridos pelo otimizador. Após isso, treinamos a função e cálculamos seu erro com base no erro quadrático médio (RMSE) desabilitando o cálculo de gradientes na rede. Foi criado um estudo com objetivo de minimizar a função objetivo, que em nosso caso é o RMSE.

<p align = 'justify'><b>Conclusão:</b> Com esse estudo, pudemos aplicar os conhecimentos de sala de aula nessa primeira etapa da disciplina, com um foco maior nas redes neurais do tipo MLP e seus usos. Aplicamos a um dataset exemplo em treinamento, analisamos a medida de "Loss" ou perda do modelo com o decorrer das épocas e utilizamos como métrica principal para o desempenho do modelo, o RMSE. Ao final foi feita uma comparação dos valores previstos pela rede com os valores reais, obtendo dados que plotados em espalhamento tinham tendencia linear, o que nos informa um bom desenpenho na predição desses valores pela rede


</p>

