# Traffic Sign Classification using fastai library
Notebooks that learns and evaluate a deep classifier of both the [Belgium](http://btsd.ethz.ch/shareddata/) and [rMASTIF](http://www.zemris.fer.hr/~kalfa/Datasets/rMASTIF/) Traffic Sign Classification datasets using the [fastai library](http://www.fast.ai/).

Here is a visualization of the Belgian traffic signs classes:

![BTS](Images/BTSC_examples.png)

And below a visualization of the rMASTIF (Croatian Traffic Signs) classes:
![rMASTIF](Images/rMASTIF_examples.png)

There is also [a SlideShare](https://www.slideshare.net/sebderhy/traffic-sign-classification-with-fastai-library-101010467) that gives high-level details and explanations. 

The state-of-the-art on these 2 datasets was mentioned in 2015 by the paper ["OneCNN"](https://www.fer.unizg.hr/_download/repository/ACPR_2015_JurisicFilkovicKalafatic.pdf). I could not find more recent and better results on one of these 2 datasets, so I assume this is the state-of-the-art. Below is their results table:

![ResultsTable](Images/ResultsTable.png)

By making simple adaptations of the [Dog Vs Cat course](http://course.fast.ai/lessons/lesson1.html) and [notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb), I was able to:
- Reach 99.4% test accuracy on the BTSC dataset, which means that the error is more than 2x smaller than the best result in the table above (Zhu et al., "Traffic sign classification using two-layer image representation", which scored 98.77% accuracy).  
- Reach 99.5% accuracy on the rMASTIF dataset. This is approximately the result was obtained by OneCNN (best result to my knowledge), but this notebook achieves it in ~3000 iterations (compared to 25K for OneCNN), and by using only the rMASTIF dataset (OneCNN trains on the German, Belgian, and rMASTIF datasets).
