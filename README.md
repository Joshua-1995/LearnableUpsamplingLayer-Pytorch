# LearnableUpsamplingLayer-Pytorch
Pytorch implementation of LearnableUpsamplingLayer (NaturalSpeech, Tan et al., 2022)

---
# Usage

``` python
'''
y : phoneme hidden sequence [N, C, T]
duration_pred : phoneme duration [N, T]
src_mask : mask of phoneme hidden sequence [N, 1, T]
'''

from model import LearnableUpsamplingLayer

lu = LearnableUpsamplingLayer
y, mel_mask = lu(y, duration_pred, src_mask) 

```
