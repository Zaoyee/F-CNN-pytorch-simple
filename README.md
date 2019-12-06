# F-CNN-pytorch-simple
 
This repo will reproduce the Figure 4 in the paper [[1]](https://arxiv.org/abs/1802.07167).

![Semantic image][logo]

[logo]: https://github.com/Zaoyee/F-CNN-pytorch-simple/blob/master/doc/image%20aim.png "Semantic image"
> [1] Long, et al. Fully Convolutional Networks for Semantic Segmentation, Figure 4.

The following figure is reproduced by this repo.

![Reproduced image][im2]

[im2]: https://github.com/Zaoyee/F-CNN-pytorch-simple/blob/master/Figs/resultsfigs.png "Reproduced image"

----

#### Re-produce the project

Before run the code, run

```
rm -f ./Figs/*.png
```

to remove all generated figures.

type `make` in terminal if you have GNU MAKE installed properlly in OS

----

The project aims to build F-CNN model in [pytorch](https://pytorch.org/).

Here, we are using [Pascal 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

The [proposal](https://github.com/Zaoyee/F-CNN-pytorch-simple/blob/master/doc/Project-2-Week-1.pdf) describes the structure in detail.
