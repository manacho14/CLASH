# Multiband galaxy morphologies for CLASH: a convolutional neural network transferred from CANDELS

This repository was made to reproduce results of our paper titled [Multiband galaxy morphologies for CLASH: a convolutional neural network transferred from CANDELS](https://arxiv.org/abs/1810.07857) (currently accepted to Publications of the Astronomical Society of the Pacific (PASP))


### Prerequisites

To reproduce our results you have to install following python3 packages:

```
- numpy
- pandas
- matplotlib
- keras
- scipy
- astropy
```


## Reproducing results

We trained a baseline model on CANDELS images and transfered it to CLASH images. In order to reproduce our results, first step is download CANDELS/CLASH data. You must have images and labels for each survey.

### CANDELS data:

Our baseline model was trained using HST images from a CANDELS GOOD-S field ([Giavalisco et al. 2004](http://adsabs.harvard.edu/abs/2004ApJ...600L..93G)), taken with WFC3 in the F160W band from Hubble Legacy Fields (HLF) Data Release 1.5 for the GOODS-S region ([HLF-GOODS-S](https://archive.stsci.edu/prepds/hlf/)). 
We used galaxies from [Kartaltepe et al. 2015](https://arxiv.org/pdf/1401.2455.pdf) catalog, selecting galaxies with
F160W magnitudes Hmag < 24.5, which corresponds to the flux limit for reliable visual morphological classifications. 
We created postage-stamp images from the GOODS-S mosaic setting the size to four times the Petrosian radius as reported in the catalog of [Guo et al. 2013](http://adsabs.harvard.edu/abs/2013ApJS..207...24G). 

### CLASH data:

For the transfer learning we used images from the CLASH Multi-Cycle Treasury program [Postman et al. 2012](https://arxiv.org/pdf/1106.3328.pdf). CLASH observed 25 clusters of galaxies in up to 16 filters, namely F225W, F275W, F336W, F390W, F435W, F475W, F606W, F625W, F775W, F814W, F850W, F105W, F110W, F125W F140W and F160W, covering the ultra violet (UV), optical (OPT) and near-infrared (NIR) regions of the spectrum. [Molino et al. 2017](https://arxiv.org/pdf/1705.02265.pdf) published accurate multiwavelength photometric catalogs for these clusters which also provide the Petrosian radius. With this data we created postage-stamp using mosaics from [MAST](https://archive.stsci.edu/prepds/clash/) for each filter separately following the same criterion for the magnitude cut and the size that we adopted for CANDELS ending up with a sample of 68, 531 galaxies.

### Training the algorithm

Once you have CANDELS/CLASH images, you must put stamps in following path './CANDELS/stamps/' or './CLASH/stamps/' and excecute train.py -dataset-, where -dataset- represent either CANDELS or CLASH. If you want to train only using CLASH data, you have to download [CANDELS weights](http://empty) in order to initialize the model.

## Testing

If you want to test the model using your own CLASH images without training a new model, you must initialize the algorithm using parameters presented in our paper, downloading them [here](http://empty).
Once parameters are downloaded, you have to place your CLASH images in './galaxies_to_predict/' and follow example provided in [Predict_example.ipynb](https://github.com/mperezcarrasco/CLASH/blob/master/Predict_example.ipynb)

## The Catalog

You can download the catalog presented in our paper [here](https://github.com/mperezcarrasco/CLASH/blob/master/Deep-CLASH.csv).

## Contributing

If you have a question do not hesitate to contact us at following [email](maperezc@udec.cl).

## Authors

* **Manuel Pérez Carrasco** - *Msc. student, department of Computer Science, University of Concepción* - [Github](https://github.com/mperezcarrasco/)
* **Guillermo Cabrera Vives** - *Assistant professor, department of Computer Science, University of Concepción* 
* **Monserrat Martinez Marín** - *Msc. student, department of Astronomy, University of Concepción*
* **Pierluigi Cerulo** - *Postdoctoral fellow, department of Astronomy, University of Concepción*
* **Ricardo Demarco** - *Assistant professor, department of Astronomy, University of Concepción*
* **Pavlos Protopapas** - *Scientific director, Institute for applied computational sciences, Harvard University*
* **Julio Godoy** - *Assistant professor, department of Computer Science, University of Concepción*
* **Marc Huertas-Company** - *Assistant professor, department of Astronomy, University Paris Diderot*
