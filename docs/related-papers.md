## Related Papers

### Toward Discovery of the Artist's Style: Learning to recognize artists by their artworks

Dataset: Rijksmuseum challenge
Using PigeoNET architecture, based on AlexNET. It seems they transferred the
 weights before actually training, but I’m not really sure.
78% of accuracy in authorship attribution over 34 classes (for some artists,
   97.5%. To others, 60.6%).
"PigeoNET was able to detect dual authorship, even though they were not 
labeled."
The network suffer greatly from noise inserted into the dataset during the
 digitalization process of the samples (camera, perspective, illumination etc).


### Visual stylometry using background selection and wavelet-HMT-based Fisher information distances for attribution and dating of impressionist paintings

Two datasets of impressionist and post-impressionist paintings from the
collections of the Van Gogh and Kröller-Muller museums.

"We divide each painting into as many non-overlapping 512  512 pixel
(approximately 2.5 inches x 2.5 inches) patches as possible, excluding the
edge of the canvas. Labeling of all painting patches into one of two subsets,
background patches are included in our analysis.
‘‘Background’’ or ‘‘Detail’’, is then done manually by a non-expert, and only
Statistical-based model created from manually extracted features using wavelet
transformations and given to a Hidden Markov Tree method.
85%~87% of accuracy in separate VG from NVG."


### Features combination for art authentication studies: brushstroke and materials analysis of Amadeo de Souza-Cardoso

"Brushstroke analysis was performed applying Gabor filter in the areas around
the keypoints detected by Scale Invariant Feature Transform (SIFT).The
characterization of the pigments was performed using hyperspectral imaging
combined with X-ray fluorescence analysis."

"The brushstroke analysis was performed using Gabor (3) strategy with an
accuracy of the 88 % (94 % true positive; 89 % true negative and 20 % false
positive answers)

"An evaluation based on visual features only should classify as Amadeo a copy
of an original painting realized nowadays; on the other hand, the assessment
based on the pigments palette only should erroneously classify a painting
realized by another artist belongs to the Amadeo’s circle."


### Large-scale Classification of Fine-Art Paintings: Learning The Right Metric on The Right Feature

1. approach: to extract visual features from images of paintings, learning a
similarity metric optimized for a given task. Having the metric learned,
we project the raw visual features into the new optimized feature space and
learn one-vs-all SVMs for the corresponding prediction task.

2. to extract different types of visual features (four types of features as
will explained next). Based on the prediction task (e.g. style) we learn the
metric for each type of feature as before. After projecting these features
separately, we concatenate them to make the final feature vector.
The classification will be based on training classifiers using these final
features.

3. projects each visual features using multiple metrics (in our experiment we
used five metrics as will be explained next) and then fuses the resulting
optimized feature spaces to obtain a final feature vector for classification.
This is an important strategy, because each one of the metric learning
approaches use a different criteria to learn the similarity measurement. By
learning all metrics individually (on the same type of feature), we make sure
that we took into account all criteria (e.g. information theory along with
neighbor hood analysis).

**GIST features** extracted from images.

CNNs used. Architecture: 4 conv layers and 3 fully connected.

Features extracted -> PCA (every feature space to dimension 512) -> classifier.

The third approach achieved the best results, and yet reduced the space
and memory requirements to 90%.


### Impressionism, Expressionism, Surrealism: Automated Recognition of Painters and Schools of Art

An image is divided into 16 equal-sized (150px^2) tiles. Multiple distinct 
features are extracted from each tile. Average vector is computed.

"A 9-way classifier that simply classifies a given painting into
one of the nine painter classes, three 3-way classifiers for the painters 
within each school of art, one 3-way classifier for the schools of art, and a 
two-stage classifier that first classifies paintings to schools
of art, and then classifies to the painter within that school. The purpose of 
using the two-stage classifier was to improve the selection and scoring of the 
relevant image features for each of the classification problems. The 
similarities between the paintings and the painters were also tested to show 
that the method can unsupervisely associate painters of the same artistic 
style."

Two-stage classifier had an accuracy of 77% over 9 different artists. 


### Seeing Behind the Camera: Identifying the Authorship of a Photograph

Dataset: 181,948 images of 41 photographers, varying resolutions.
CNNs! CaffeNet, Hybrid-CNN, PhotographerNET.
Train/valid split: (.9, .1)
Hybrid-CNN had 74% of accuracy.