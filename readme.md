#Introduction

we are going to use sigular vector decomposition to extract information from document.

This class LSA and SVD method only applied for single document. 

## sample
...
(CNN)For the second time during his papacy, Pope Francis has announced a new group of bishops and archbishops set to become cardinals -- and they come from all over the world.

...

@highlight

The 15 new cardinals will be installed on February 14

@highlight

They come from countries such as Myanmar and Tonga

@highlight

No Americans made the list this time or the previous time in Francis' papacy
...

## steps
* we read a sample file from cnn/sample from cnn/stories;
* split the sample file into the document part and highlight parts;
* create an vocabulary x sentence matrix from document;
* split documents by sentences, and read one by one by counting unique words in the sentence and put that vector into matrix;
* perform SVD on the matrix;
* TBD

# Result and Performance

## train the model with all words

## train the model after filtering all stop words from NLTK

## traditional SVD

# ref

https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html

SVD in np

'''
Reconstruction based on full SVD, 2D case:

>>>
>>> u, s, vh = np.linalg.svd(a, full_matrices=True)
>>> u.shape, s.shape, vh.shape
((9, 9), (6,), (6, 6))
>>> np.allclose(a, np.dot(u[:, :6] * s, vh))
True
>>> smat = np.zeros((9, 6), dtype=complex)
>>> smat[:6, :6] = np.diag(s)
>>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
True

Reconstruction based on reduced SVD, 2D case:

>>>
>>> u, s, vh = np.linalg.svd(a, full_matrices=False)
>>> u.shape, s.shape, vh.shape
((9, 6), (6,), (6, 6))
>>> np.allclose(a, np.dot(u * s, vh))
True
>>> smat = np.diag(s)
>>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
True
'''