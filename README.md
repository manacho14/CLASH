# Multiband galaxy morphologies for CLASH: a convolutional neural network transferred from CANDELS

This repository was made to reproduce results of our paper titled "Multiband galaxy morphologies for CLASH: a convolutional neural network transferred from CANDELS"


### Prerequisites

What things you need to reproduce our results are following python3 packages

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
F160W magnitudes Hmag < 24.5. This is the same limit used in KA15, who show that it corresponds to the flux limit for reliable visual morphological classifications. We created postage-stamp images from the GOODS-S mosaic setting the size to four times the Petrosian radius as reported in the catalog of [Guo et al. 2013](http://adsabs.harvard.edu/abs/2013ApJS..207...24G). 


For the transfer learning sample we used images from
the CLASH Multi-Cycle Treasury program (Postman et al. 2005). CLASH observed 25 clusters of galaxies at redshifts
0.15 < z < 0.9 with WFC3 over a period of 3 years, in up to
16 filters, namely F225W, F275W, F336W, F390W, F435W,
F475W, F606W, F625W, F775W, F814W, F850W, F105W,
F110W, F125W F140W and F160W, covering the ultra violet (UV), optical (OPT) and NIR regions of the spectrum.
Molino et al. (2017) published accurate multiwavelength
photometric catalogs for these clusters which also provide
the Petrosian radius. We created postage-stamp images for
each filter separately following the same criterion for the
magnitude cut and the size that we adopted for CANDELS
ending up with a sample of 68, 531 galaxies.




### CLASH data:

[Molino et al. 2017](https://arxiv.org/pdf/1705.02265.pdf)

## Testing results

In order to test 

```
Give an example
```


## Contributing

If you have a question do not hesitate to contact us at following [email](maperezc@udec.cl).

## Authors

* **Manuel Pérez Carrasco** - *Msc. student, department of Computer Science, University of Concepción* - [Github](https://github.com/mperezcarrasco/)
* **Guillermo Cabrera Vives** - *Associated professor, department of Computer Science, University of Concepción* 
* **Monserrat Martinez Marín** - *Msc. student, department of Astronomy, University of Concepción*
* **Pierluigi Cerulo** - *Postdoctoral fellow, department of Astronomy, University of Concepción*
* **Ricardo Demarco** - *Associated professor, department of Astronomy, University of Concepción*
* **Pavlos Protopapas** - *Scientific director, Institute for applied computational sciences, Harvard University*
* **Julio Godoy** - *Associated professor, department of Computer Science, University of Concepción*
* **Marc Huertas-Company** - *Assistant professor, department of Astronomy, University Paris Diderot*
