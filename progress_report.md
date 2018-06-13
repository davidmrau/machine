### What we did so far:

- Extract parameters for all layers.

- Generate heatmaps for all layers.

- Run basic statistical analysis.

- Compare distributions of parameters within set of guided/baseline models. For all layers separately.

- Compare distributions of parameters of guided vs. baseline models. For all layers separately.

- Train classifier on rows of weight matrices as input. For every layer separately.

- Replaced parts of the network of guided vs. baseline models respectively.


# Findigs:
 


Compare distributions of parameters

- **decoder.embedding.weight**, **decoder.attention.method.out.weight** show biggest difference in distribution. This holds for both GRU and LSTM.

This can be seen by inspecting the parameter distribution plots:

- **decoder.attention.method.out.weight** for baseline model accumulated around zero. For guided model less weights arounds zero and weights are more spread out.

![GRU attention](machine/images/distribution/hist\_Baseline\_GRU\_decoder.attention.method.out.weight.png)

- **decoder.embedding.weight** for baseline model values accumulate around 0 and +-0.25, whereas for the guided model values are more spread out. Peak around +-0.1.

![GRU embedding](machine/images/distribution/hist\_Baseline\_GRU\_decoder.embedding.weight.png)

Heatmaps


- **decoder.embedding.weight** for both models neuron two seems to encode imporant information. For the baseline model the weights from neuron 3-11 are around zer. For the Guided model those neurons the values differ a lot and seem to encode information which the baseline model was not able to encode.

![GRU embedding heat](machine/images/heatmap/Guided\_GRU\_3\_decoder.embedding.weight.png)

![GRU embedding heat](machine/images/heatmap/Baseline\_GRU\_3\_decoder.embedding.weight.png)




- **decoder.attention.method.mlp.weight** more noisy for Guided vs. Baseline


![GRU embedding heat](machine/images/heatmap/Guided\_LSTM\_4\_decoder.attention.method.mlp.weight.png)


![GRU embedding heat](machine/images/heatmap/Baseline\_LSTM\_4\_decoder.attention.method.mlp.weight.png)


- **decoder.out.bias** shows also a big difference in distribution of guided vs. baseline but also within each class. This also holds fo GRU and LSTM.



Classification
-

- layers for which the binary classification (guided/baseline) is >= 0.9

LSTM:

encoder.rnn.weight\_w\_hi, encoder.rnn.weight\_w\_hf, encoder.rnn.weight\_w\_hc, encoder.rnn.weight\_w\_ho, decoder.rnn.weight\_w\_ii, decoder.rnn.weight\_w\_if, decoder.rnn.weight\_w\_io, decoder.rnn.weight\_w\_hi, decoder.rnn.weight\_w\_hf, decoder.rnn.weight\_w\_ho, decoder.attention.method.mlp.weight

GRU: 

encoder.rnn.weight\_w\_hr,
encoder.rnn.weight\_w\_hi,
encoder.rnn.weight\_w\_hn,
decoder.rnn.weight\_w\_ir,
decoder.rnn.weight\_w\_ii,
decoder.rnn.weight\_w\_hr,
decoder.rnn.weight\_w\_hi,
decoder.attention.method.mlp.weight,


Transfer learning
-

Take **decoder of guided model**, freeze weights and place it into **Baseline model**. Retrain hybrid model.

- Acc heldout\_tables : **0.9010** (before baseline: 0.0000, guided: 0.8490)

- Acc new\_compositions: **0.8438** (before baseline: 0.0417, guided: 0.6875)




