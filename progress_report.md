### What we did so far:

- Extract parameters for all layers.

- Generate heatmaps for all layers.

- Run basic statistical analysis.

- Compare distributions of parameters within set of guided/baseline models. For all layers separately.

- Compare distributions of parameters of guided vs. baseline models. For all layers separately.

- Train classifier on rows of weight matrices as input. For every layer separately.

- Replaced parts of the network of guided vs. baseline models respectively.

 


Compare distributions of parameters
-

- **decoder.embedding.weight**, **decoder.attention.method.out.weight** show biggest difference in distribution. This holds for both GRU and LSTM.

This can be seen by inspecting the parameter distribution plots:

- **decoder.attention.method.out.weight** for baseline model accumulated around zero. For guided model less weights arounds zero and weights are more spread out.

![GRU attention](images/distribution/hist\_Baseline\_GRU\_decoder.attention.method.out.weight.png)

- **decoder.embedding.weight** for baseline model values accumulate around 0 and +-0.25, whereas for the guided model values are more spread out. Peak around +-0.1.

![GRU embedding](images/distribution/hist\_Baseline\_GRU\_decoder.embedding.weight.png)

Heatmaps
-

- **decoder.embedding.weight** . The first three rows decode for the tokens: 1. \<eos\>\<unk\> 2. \<pad\> 3. \<sos\>. F.or both models \<sos\> shows a really strong pattern. For the baseline model the weights from neuron 3-11 are around zero. For the Guided model those values differ a lot and seem to encode information.

![GRU embedding heat](images/heatmap/Guided\_GRU\_3\_decoder.embedding.weight.png)

![GRU embedding heat](images/heatmap/Baseline\_GRU\_3\_decoder.embedding.weight.png)



- **decoder.attention.method.mlp.weight** for some rows the guided model shows distinct lines which can't be found in any baseline heatmap

![GRU embedding heat](images/heatmap/Guided\_LSTM\_4\_decoder.attention.method.mlp.weight.png)


![GRU embedding heat](images/heatmap/Baseline\_LSTM\_4\_decoder.attention.method.mlp.weight.png)


- **decoder.out.bias** shows also a big difference in distribution of guided vs. baseline but also within each class. This also holds fo GRU and LSTM.



Classification
-

For the classification both classes (guided/baseline) were whitened independently. This was done in order to find only non trivial differences in the two classes, such as mean or standart deviation of the parameters. As inputs for the training we used rows of the weight matrices as instances.

- Layers for which the binary classification (guided/baseline) acc is \>= 0.9 on a network with 10 hidden layers, 2 outputs.

LSTM:




| Parameter | avg f1-score | feature size | training instances|
| ---:|:----------------- |:---------------- |:------------------------ | 
|encoder.embedding.weight |0.40 | 16 | 152 | 
| encoder.rnn.weight\_w\_ii | 0.79 | 16 | 4096 | 
| encoder.rnn.weight\_w\_if | 0.84 | 16 |4096 |  
| encoder.rnn.weight\_w\_ic | 0.49 | 16 |4096 |  
|encoder.rnn.weight\_w\_io | 0.79 | 16 |4096 |  
|encoder.rnn.weight\_w\_hi | 0.99| 512 |4096 |  
|encoder.rnn.weight\_w\_hf | 0.97| 512 |4096 |  
|encoder.rnn.weight\_w\_hc | 0.93| 512 |4096 |  
|encoder.rnn.weight\_w\_ho | 0.98| 512 |4096 |  
| decoder.rnn.weight\_w\_ii | 0.90 | 512 | 4096 | 
| decoder.rnn.weight\_w\_if | 0.88 | 512 |4096 |  
| decoder.rnn.weight\_w\_ic | 0.94 | 512 |4096 |  
|decoder.rnn.weight\_w\_io | 0.79 | 512 |4096 |  
|decoder.rnn.weight\_w\_hi | 0.99| 512 |4096 |  
|decoder.rnn.weight\_w\_hf | 0.97| 512 |4096 |  
|decoder.rnn.weight\_w\_hc | 0.93| 512 |4096 |  
|decoder.rnn.weight\_w\_ho | 0.98| 512 |4096 |  
|decoder.embedding.weight | 0.50 |512 |88 | 
|decoder.attention.method.mlp.weight | 0.98 | 1024 | 4096 |  






Transfer learning
-


- Did swapping with baseline and AG models 1 from model zoo.

- The letter represents the model being retrained (A for for a pre-trained AG model, again being retrained with AG, B for a pre-trained baseline model, again being retrained without AG). The name of the component represents the component taken from the opposite model and of which the weights have been frozen.




| Model | Heldout compositions | Heldout inputs | Heldout tables | New compositions |
| ---:| :------------------- |:-------------- |:-------------- |:---------------- |
|  A-decoder| 0.4219 | 0.3250 | 0.2240 | 0.1875 |
| A-encoder | 0.4531 | 0.3000 | 0.1094 | 0.0625 |
| B-decoder | 0.1719 | 0.1750 | 0.0469 | 0.0312 |
| B-encoder | 0.3125 | 0.3500 | 0.1198 | 0.000 |
|A-decoder.embedding | 0.9375 | 0.9750 | 0.5417 | 0.25|
|A-encoder.embedding | 1 | 1 | 0.8542 | 0.6875|
| B-decoder.embedding |0.1719|0.1750|0.0469|0.0000 |
| B-encoder.embedding | 	0.1719	| 0.1750 | 0.0521 | 0.0625 |
| Baseline | 0.2500 |	0.2000	| 0.0417 |	0.0000|
| Guided | 1.0000|1.0000 |	0.8490|	0.6875|


Conclusions
-

- Token 2 (\<sos\>) in decoder.embedding seems to be very different from all other token embeddings in the weight heatmap

- The decoder token embeddings for the actual bitstrings seem to be different for AG vs baseline in weight heatmap, and this also shows in the weight histogram comparison 

- The encoder embeddings are not important for finding a compositional solution, because an AG model can be properly retrained with frozen encoder embeddings from the baseline

- The decoder embeddings do seem to be important, since a retrained AG model can not reach the same performance with frozen decoder embeddings from the baseline 

- The encoder does seem important for a compositional solution, since a frozen encoder from an AG model placed into a baseline model and retrained without AG beats all other baseline models




