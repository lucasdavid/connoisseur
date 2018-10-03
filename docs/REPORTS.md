## Reports

### van Gogh's

van Gogh's dataset is composed by 264 train samples and 67 test samples,
discriminated by the non van Gogh (nvg) and van Gogh (vg) labels.

#### Experimenting on Surpassing the Baseline

| classifier | extracted patches | patch acc. | acc. | notes |
| --- | --- | --- | --- | --- |
| vgg19-fc2 imagenet svm (Folego, 2016) | all (~187) 224px | ? | 94.0299% |
| vgg19-fc2 imagenet svm | random 224px | 89.3134% | 95.5224% |
| histogram 64 bins svm | random 299px | 62.6866% | 62.6866% | over-fitting to all vangogh's |
| histogram 64 bins svm | min-gradient 299px | 62.6866% | 62.6866% | over-fitting to all vangogh's |
| histogram 64 bins svm | max-gradient 299px | 62.6866% | 62.6866% | over-fitting to all vangogh's |
| InceptionV3-mixed4 imagenet svm | min-gradient 299px | 90.7463% | 94.0299% |
| InceptionV3-mixed7 imagenet svm | min-gradient 299px | 90.4179% | 94.0299% |
| **InceptionV3-global_avg imagenet svm** | min-gradient 299px | 88.6866% | 95.5224% |
| InceptionV3-global_avg imagenet svm | max-gradient 299px | 88.7164% | 92.5373% |
| InceptionV3-global_avg imagenet svm | random 299px | 88.7164% | 95.5224% |
| fine-tuned inception-global_avg svm  | min-gradient 299px | 90.6866% | 94.0299% |
| Xception-avg_pool imagenet pca(0.95) svm | random 299px | 89.0299% | 94.0299% |
| ResNet50-avg_pool imagenet pca(0.99) svm | random 299px | 88.8806% | 91.0448% |
| densenet-40 softmax | random 32px | 85.3881% | 91.0448% |
| densenet-40 svm | random 32px | 86.6343% | 88.0597% |
| densenet-264 softmax | random 32px | 87.0149% | 91.0448% | only 4 features after PCA |


#### Experimenting on Reducing the Number of Patches Used

Hypothesis: scores can be kept high while reducing the number of patches.

| classifier | extracted patches | patches valid acc. | patches test acc. | test acc. |
| --- | --- | --- | --- | --- |
| vgg19-fc2 imagenet svm (Folego, 2016) | all (~187) | - | ? | 94.0299% |
| vgg19-flatten imagenet svm  | 50 random | -  | 89.7114% | 95.5224% |
| vgg19-flatten imagenet svm  | 20 random | - | 88.9540% | 91.0448% |
| vgg19-fc2 imagenet pca(0.95) svm | 50 random | - | 89.3134% | 95.5224% |
| vgg19-fc2 imagenet pca(0.95) svm | 20 random | - | 87.9104% | 91.0448% |
| vgg19-fc2 imagenet pca(0.99) svm | 20 random | - | 87.9851% | 92.5373% |
| inception-avg_pool imagenet pca(0.99) svm | 50 random | - | 89.0448% | 95.5224% |
| inception-avg_pool imagenet pca(0.99) svm | 20 random | - | 89.7014% | 95.5224% |
| inception-avg_pool imagenet pca(0.95) svm | 20 random | - | 90.1493% | 97.0149% |
| inception-avg_pool imagenet pca(0.99) svm | 10 random | - | 88.9552% | 94.0299% |
| inception-avg_pool imagenet pca(0.95) svm |  5 random | - | 88.8955% | 92.5373% |
| inception-avg_pool imagenet pca(0.95) svm |  2 random | - | 86.9850% | 91.0448% |
| inception-avg_pool imagenet pca(0.99) svm |  1 random | - | 86.4776% | 83.5821% |
| densenet softmax | all (~900) | 82.22% | - | - |
| densenet softmax | 500 random | 82.42% | - | - |
| densenet softmax | 200 random | 87.08% | 86.6343% | 88.0597% |
| densenet softmax | 50 random | 87.08% | - | 92.5373% |
| densenet pca(0.99) svm | 200 random | 87.08% | 85.3880% | 91.0448% |

#### InceptionV3, PCA and SVM

So far, the method responsible for best classification accuracy in the test
set is as follows:

![Diagram of best scoring pipeline](../assets/vangogh-inceptionv3-svm.png)

1. `1-extract-patches.py` is used to extract 50 random patches with sizes
(299, 299, 3) from each sample in train and test folders.
2. `4a-embed-patches.py` is executed. Using InceptionV3 architecture and
its weights trained over imagenet dataset, patches are embedded to a lower
dimensional space (cut-point is `GlobalAveragePooling2d(name='avg_pool')` layer).
3. `5-train-top-classifier` trains a PCA --> SVM pipeline that classifies
patches according to their labels.
4. `6-evaluate-fusion` fuses the answers from the model trained above over the
test dataset using each strategy (e.g. sum, mean, farthest, most_frequent) and
reports results. Best values are shown bellow:

   ```
   score using raw strategy: 90.9971% 
   Confusion matrix:
        nvg   vg
   nvg 1821  279
    vg   59 1191
     
   class-balanced test score using sum strategy: 96.4286%

   Confusion matrix:
       nvg  vg
   nvg 39    3
    vg  0   25

   samples incorrectly classified: nvg_10658644 nvg_10500055 nvg_18195595
   ```

#### DenseNet264, PCA and SVM

1. `1-extract-patches.py` is used to extract 50 random patches with sizes
(32, 32, 3) from each sample in train and test folders.
2. `2-train-network.py` trains `DenseNet264` over the extracted patches.
3. `3-optional-generate-network-predictions.py` shows the following results:

   ```
   valid patches score: 0.846538461538
   Confusion matrix:
        nvg   vg
   nvg 1403  197
    vg  202  798

   valid score using farthest strategy: 0.942307692308
   Confusion matrix:
       nvg   vg
   nvg  32    0
    vg   3   17

   test patches score: 0.842089552239
   Confusion matrix:
        nvg   vg
   nvg 1746  354
    vg  175 1075

   test score using farthest strategy: 0.925373134328
       nvg   vg
   nvg  39    3
    vg   2   23

   samples incorrectly classified: nvg_10658644 nvg_9780042 nvg_6860814
                                   vg_17177301 vg_33566806
   ```

4. `4a-embed-patches.py` is executed. Using DenseNet264 architecture trained
in the previous step, patches are embedded to a lower dimensional space
(cut-point is `GlobalAveragePooling2d(name='avg_pool')` layer).
5. `5-train-top-classifier` trains a PCA --> SVM pipeline that classifies
patches according to their labels. One interesting result here is that PCA
reduces the network output to only 4 dimensions, making the rest of the
classification pipeline very efficient.
6. `6-evaluate-fusion` fuses the answers from the model trained above over the
test dataset using each strategy (e.g. sum, mean, farthest, most_frequent) and
reports results. Best values are shown bellow:

   ```
   class-balanced test patches score: 86.1943%

   Confusion matrix:
         nvg   vg
    nvg 1878  222
     vg  213 1037

   class-balanced test score using mean strategy: 90.4286%

   Confusion matrix:
         nvg   vg
    nvg   39    3
    vg     3   22

   samples incorrectly classified: nvg_10658644 nvg_9780042 nvg_6860814
                                   vg_17177301 vg_33566806 vg_9463608
   ```

#### Further Testing

The previous classifiers performed well when applied to the test paintings.
How well would they perform on recaptures of those same paintings and
different van Gogh paintings?

When gathering the recaptures datasets, I reached for closer resolutions
to the ones presented in vgdb2016 whenever available. Unfortunately, while the
vgdb2016 contains huge files, recaptures from google were recurrently smaller
than (2000px, 2000px).

```shell
general statistics on test paintings in vgdb_2016:
  min sizes: (1016 px 1044 px)
  avg sizes: (3115 px 2805 px)
  max sizes: (7172 px 7243 px)
  min  area: 1419840 px**2
  avg  area: 10047495 px**2
  max  area: 45355666 px**2
```

In order to attempt to measure this difference in the process, two strategies
are considered:

1. none: images were used as they were when retrieved.
2. resized: images are resized to either match their correspondent's width
          in vgdb2016 (if they are recaptures) or the average width in
          vgdb2016 (if they are unseen images).


##### Recaptures from van Gogh Museum

Recaptures from images in vgdb2016 test set were retrieved from the van Gogh museum website.

| clf | extracted patches | processing | patch confusion m. | best | confusion m. | acc. |
| --- | --- | --- | --- | --- | --- | --- |
| folego's | all | none | 0 0<br>1095 136 | farthest | 0 0<br>6 2 | **25%** |
| svm | random | none | 0 0<br>288 112 | frequent | 0 0<br>6 2 | **25%** |
| svm | random | resized | 0 0<br>302 98 | mean, farthest, frequent | 0 0<br>7 1 | 13% |
| svm | min-grad | none | 0 0<br>293 107 | mean, farthest, frequent | 0 0<br>7 1 | 13% |
| svm | min-grad | resized | 0 0<br>317 83 | mean, farthest | 0 0<br>7 1 | 13% |
| densenet264 | random | none | 0 0<br>390 10 | farthest | 0 0<br>8 0 | 0% |
| densenet264 | random | resized | 0 0<br>373 27 | farthest | 0 0<br>7 1 | 13% |

##### van Gogh Museum Unseen

Images of paintings that do not appear in vgdb2016 were retrieved from the
vangogh museum website.

| clf | extracted patches | transf. | patch confusion m. | best | confusion m. | acc. |
| --- | --- | --- | --- | --- | --- | --- |
| folego's | all | none | 1782  289<br>5239 1413 | farthest | 9  2<br>24 10 | 56% |
| svm | random | none | 474 76<br>941 759 | mean, farthest | 10 1<br>20 14 | 66% |
| svm | random | resized | 469 81<br>979 721 | frequent | 11 0<br>19 15 | **72%** |
| svm | min-grad | none | 466 84<br>941 759 | farthest | 11 0<br>20 14 | 71% |
| svm | min-grad | resized | 472 78<br>956 744 | farthest | 10 1<br>19 15 | 68% |
| densenet264 | random | none | 473 77<br>1512 188 | farthest | 8 3<br>26 8 | 48% |
| densenet264 | random | resized | 497 53<br>1530 170 | farthest | 10 1<br>26 8 | 57% |


##### Recaptures from Multiple Places

Recaptures from images in vgdb2016 test set were retrieved from multiple
placed, found using google image search.

| clf | extracted patches | transf. | patch confusion m. | best | confusion m. | acc. |
| --- | --- | --- | --- | --- | --- | --- |
| folego's | all | none | 13967 3503<br>6189 7768 | farthest | 79 6<br>21 36 | 78% |
| svm | random | none | 3883 367<br>969 1881 | frequent | 82 3<br>19 38 | **82%** |
| svm | random | resized | 3697 553<br>921 1929 | farthest | 78 7<br>19 38 | 79% |
| svm | min-grad | none | 3836 414<br>927 1923 | mean, frequent | 82 3<br>19 38 | **82%** |
| svm | min-grad | resized | 3642 608<br>905 1945 | frequent | 79 6<br>19 38 | 80% |
| densenet264 | random | none | 3837 413<br>1890 960 | mean | 81 4<br>40 17 | 63% |
| densenet264 | random | resized | 3910 340<br>1887 963 | farthest | 79 6<br>34 23 | 67% |

###### Combining Recaptures

Recaptures were grouped by the painting the contain and `frequent` is used
to form a final classification for the painting.

```
class-normalized accuracy: 86.68%
Confusion matrix:
 37 (97%)  1
  6       19 (76%)
samples incorrectly classified: nvg_10500055 vg_9103139 vg_9386980 vg_9387502
                                vg_9414279 vg_9421984 vg_9463012 
```

Adding the paintings from vangogh2016/test as recaptures:

```
class-normalized accuracy: 92.43%
Confusion matrix:
39 (93%)  3
 2       23 (92%)
samples incorrectly classified: nvg_10500055 nvg_10658644 nvg_18195595
                                vg_9103139 vg_9414279
```

#### Training over vgdb2016 + recaptures

Repetitions of the paintings in vgdb2016 train were downloaded from google and the
InceptionV3+PCA+SVM was used. Bellow are the results over the test sets.

###### vgdb2016 Test

```shell
score using raw strategy: 91.4086% 
Confusion matrix:
1803  297
  38 1212

Combining patches:
class-balanced score using sum/mean strategy: 96.429% 
Confusion matrix:
39  3
 0 25 
samples incorrectly classified: nvg_18195595 nvg_10658644 nvg_10500055
```

###### Recaptures from Multiple Places 

```shell
score using raw strategy: 0.80803 
Confusion matrix:
3802  448
 915 1935

Combining patches:
score using sum strategy: 0.85211 
Confusion matrix:
81  4
17 40
 
samples incorrectly classified:  nvg_18195595-0 vg_9463012-0 vg_9386980-1
                                 vg_9414279-2 vg_9413420-0 vg_9413420-1 vg_9103139-2
                                 vg_9103139-1 nvg_10500055-0 vg_9110201-1
                                 vg_9387502-1 vg_9414279-0 vg_9387502-3 vg_9103139-0
                                 vg_9414279-1 vg_9106795-1 nvg_10500055-1
                                 vg_9386980-0 nvg_9780042-2 vg_9378884-3 vg_9103139-3 

Combining recaptures:
score using sum/mean strategy: 0.88889
Confusion matrix:
37  1
 6 19
 
samples incorrectly classified:  nvg_10500055 vg_9103139 vg_9386980 vg_9387502
                                 vg_9413420 vg_9414279 vg_9463012 
```

###### vgdb2016 + Recaptures from Multiple Places

```shell
score using raw strategy: 0.83875 
Confusion matrix:
5619  731
 946 3104
 
Combining patches:
score using farthest strategy: 0.9038461538461539 
Confusion matrix:
121   6
 14  67 

samples incorrectly classified: vg_9414279-0 nvg_10500055-1 vg_9110201-1
                                vg_9103139-1 vg_9386980-0 vg_9413420-0
                                nvg_18195595-3 nvg_9780042-2 vg_9103139-2
                                vg_9103139-3 vg_9387502-1 vg_9506505-0 vg_9106795-1
                                nvg_10658644-1 nvg_10500055-0 vg_9414279-2
                                vg_9387502-3 vg_9378884-3 vg_9414279-1
                                nvg_10500055-2

Combining recaptures:
score using farthest strategy: 0.9552238805970149 
Confusion matrix:
40  2
 1 24
 samples incorrectly classified: nvg_10500055 nvg_10658644 vg_9414279 
```

---

### Painter by Numbers

Containing many paintings from 1564 painters, this dataset was made available
in Kaggle's [Painter-by-Numbers](https://www.kaggle.com/c/painter-by-numbers)
competition. We also have access to meta-data associated with the paintings
(e.g. style, genre and year created).

While we can interpret the training phase as a multiclass problem, the test
phase consists in deciding whether or not two paintings belonging to a same
artist.

Score is computed by ROC AUC between an estimated probabilities and the actual
label (0.0 or 1.0).

#### InceptionV3 Gram-matrix, Softmax

All samples from 100 first painters.
Process paintings were passed through a pre-trained network,
the gram matrix was computed at a determined cut-point and fed
to a softmax classifier.
						
| phase/method | vgg19,conv1_block1 | vgg19,conv2_block1 | vgg19,conv3_block1 | inception,predictions |
| --- | --- | --- | --- | --- |
| train	| 0.3438 | 0.1524 | 0.0564 | 0.2985 |
validation | 0.3177 | 0.1381 | 0.0718 | 0.2514 |
						
Best results for gram-matrix were given by `vgg199 conv1_block1`, but are
extremely bellow the results obtained when using the regular feed-forward
network.

#### Siamese InceptionV3, PCA, SVC, Equal Joint

The 100 first painters (sorted by their hash code) were considered and the
pipeline described bellow.

![Diagram of Siamese InceptionV3, PCA, SVC, Equal Joint](../assets/pbn-inception-svm-equals.png)

We can see from the train confusion matrix and the test report bellow that the
model performed well for the 100 artists selected in training, but clearly
overfitted the data and missed many samples associated with the label
`same-artist`.

![Confusion matrix for train set](../assets/pbn-100-train-cm.png)

```
farthest:
  score: 0.70295719844
  confusion-matrix:
                        different-painters same-painter
    different-painters            15289495      6339202
          same-painter              170802       116548

frequent:
  score: 0.892038742206
  confusion-matrix:
                        different-painters same-painter
    different-painters            19477643      2151054
          same-painter              215030        72320
```

#### Siamese Fine-tuned InceptionV3, Custom Joint

![Diagram of Siamese Fine-tuned InceptionV3, Dot Joint](../assets/pbn-inception-softmax-dot.png)

1. `1-extract-patches.py` is used to extract 50 random patches with sizes
(299, 299, 3) from each sample in train and test folders.
2. `2-train-network.py` fine-tunes InceptionV3 (transferred from imagenet)
to the train dataset, associating patches to their respective painters.
4. `3a-generate-network-answers` feed-forwards each test painting's test
through the network trained in the previous step. Let `y` be the fined-tuned
InceptionV3 and `a` and `b` be paintings in a pair described in
`submission_info.csv`:
   - **Equal Joint** Patches probabilities are fused using each strategy,
     leaving us with the probabilities of a work belonging to each one of the
     1584 painters. The probability of these being of a same painting is
     computed by `argmax(y(a)) == argmax(y(b))`. Results are shown bellow:

     ```
     roc auc score using mean strategy: ?

     Confusion matrix:

                         different-painters same-painter
     different-painters            21593154        35543
           same-painter              214767        72583
     ```

   - **Dot Joint** patches probabilities are fused using `mean` strategy,
     leaving us with the probabilities of a work belonging to each one of the
     1584 painters. The probability of these being of a same painting is
     computed by `y(a).dot(y(b))`. Results are shown bellow:

     ```
     roc auc score using mean strategy: 0.902789501328

     Confusion matrix:

                         different-painters same-painter
     different-painters            21627458         1239
           same-painter              261844        25506
     ```

   - **Pearsonr Joint** patches probabilities are fused using `mean` strategy,
     leaving us with the probabilities of a work belonging to each one of the
     1584 painters. The probability of these being of a same painting is
     computed by `scipy.stat.pearsonr(y(a), y(b))[0]`. Results are shown bellow:

     ```
     roc auc score using mean strategy: 0.880932204226

     Confusion matrix:

                         different-painters   same-painter
     different-painters     21588252 (100%)    40445  (0%)
           same-painter       207041  (72%)    80309 (28%)
     ```

#### Siamese Fine-tuned InceptionV3, Embedding Dense Layers, l^2 Joint

The limbs of the network trained in [Siamese Fine-tuned InceptionV3, Custom Joint](#siamese-fine-tuned-inceptionv3-custom-joint)
are used to compose a new network (frozen weights), illustrated in the diagram
bellow:

![Diagram of Siamese Fine-tuned InceptionV3, l^2 Joint](../assets/pbn-siamese-inception-1584-l2-contrastive.png)

```
roc auc: 0.831379516462
accuracy normalized by class-frequency: 73.5%

Confusion matrix:

                    different-painters   same-painter
different-painters      17965397 (83%)  3663300 (17%)
      same-painter        103340 (36%)   184010 (64%)
```


#### Siamese Fine-tuned InceptionV3, Embedding Dense Layers, Sigmoid Joint

The limbs of the network trained in [Siamese Fine-tuned InceptionV3, Custom Joint](#siamese-fine-tuned-inceptionv3-custom-joint)
are used to compose a new network (frozen weights), illustrated in the diagram
bellow:

![Diagram of Siamese Fine-tuned InceptionV3, Sigmoid Joint](../assets/pbn-siamese-inception-1584-sigmoid.png)


```
roc auc using mean strategy: 0.865582232769
accuracy normalized by class-frequency: 78%

Confusion matrix:

                    different-painters   same-painter
different-painters      16418044 (76%)  5210653 (24%)
      same-painter         56919 (20%)   230431 (80%)
```


##### Siamese Multi-label Fine-tuned InceptionV3, Embedding Dense Layers, Sigmoid Joint

![Diagram of Siamese Multi-label Fine-tuned Multi-label InceptionV3, Sigmoid Joint](../assets/pbn-siamese-inception-multilabel-1763-sigmoid.png)

```
roc auc score using mean strategy: 0.913406464881
accuracy normalized by class-frequency: 80%

Confusion matrix:

                    different-painters   same-painter
different-painters      20069328 (93%)  1559369  (7%)
      same-painter         95072 (33%)   192278 (67%)
```


##### Siamese Multi-branch Fine-tuned InceptionV3, Embedding Dense Layers, Sigmoid Joints

![Diagram of Siamese Multi-branch Fine-tuned Multi-label InceptionV3, Sigmoid Joint](../assets/pbn-siamese-inception-multiple-outputs.png)

| limb | branches used | embedding units | roc auc | acc | confusion matrix diag. |
| --- | --- | --- | --- | --- | --- |
| InceptionV3 softmax | artist, style, genre | 1024, 256, 256 | .898 | .914 | .9176 .6295 |
| InceptionV3 sigmoid | artist, style, genre | 1024, 256, 256 | .884 | .878 | .8802 .6769 |

```{r echo=FALSE, eval=FALSE}
| InceptionV3 sigmoid | artist, style, genre | 2048, 256, 128 | ? | ? | ? |
| InceptionResNetV2 sigmoid | artist, style, genre | 2048, 256, 128 | ? | ? | ? |
```
