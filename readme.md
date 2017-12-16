Bottom-Up and Top-Down Attention for Visual Question Answering

This is a implementation of the VQA system described in "Bottom-Up and
Top-Down Attention for Image Captioning and Visual Question Answering"
(https://arxiv.org/abs/1707.07998) and "Tips and Tricks for Visual
Question Answering: Learnings from the 2017 Challenge"
(https://arxiv.org/abs/1708.02711). The papers describe the winning
entry of VQA 2017 challenge.

Our implementaion follows the overall structure of the paper but with
the following simplifications:

1. We don't use extra data from Visual Genome.
2. We use only a fixed number of objects per image (K=36).
3. We use a simple, single stream classifier without pretraining.
4. We use simple ReLU activation instead of gated tanh.

The first two points greatly reduce the training time. Our
implementation takes around 200s per epoch on a single Titan Xp while
the one described in the paper takes 1 hour per epoch.

The third point is simply because we feel the two stream classifier
and pretraining in the original paper is over-complicated and not
necessary.

For activation, we tried gated tanh but cannot make it work. We also
tried gated linear unit (GLU) and in fact it works better than
ReLU. Eventually we choose ReLU due to its simplicity and the gain
from using GLU is too small to jusitify the fact that GLU doubles the
number of parameters.

With these simplification we would expect the performance to drop. For
reference, the best result on validation set reported in the paper is
63.15. The reported result without extra data from visual genome is
62.48, the result using only 36 objects per image is 62.82, the result
using two steam classifier but not pretrained is 62.28 and the result
using ReLU is 61.63. These numbers are cited from the Table1 of the
paper: "Tips and Tricks for Visual Question Answering: Learnings from
the 2017 Challenge". With all the above simplification aggregated, our
first implementation got around 59-60 on validation set.

To shrink the gap, we added some easy but powerful
modifications. Including:

1. Add dropout to alleviate overfitting problem
2. Double the number of neurons
3. Add weight normalization (BN seems not working well here)
4. Switch to Adamax optimzer
5. Gradient clipping

These small modifications bring the number back to ~62.80.  We further
change the concatenation based attention module in the original paper
to a projection based module. This new attention module is inspired by
the paper "Modeling Relationships in Referential Expressions with
Compositional Modular Networks"
(https://arxiv.org/pdf/1611.09978.pdf).  but it is slightly more
complicated than that (implemented in attention.NewAttention).  With
the help of this new attention, we boost the performance to ~63.54,
surpassing the reported best result with no extra data and less
computation cost.

However, the purpose this open source project is not to beat the
original papers. In fact, we were thinking about integrating object
detection with VQA and were very glad to see that Peter Anderson and
Damien Teney et al. had done that beautifully. We hope this clean and
efficient implementation can serve as the new baseline for future VQA
explorations.

Finally, this is part of a project done at CMU for course 11777 and
a joint work with Alex Xiao and Henry Huang.


Usage:

TODO
