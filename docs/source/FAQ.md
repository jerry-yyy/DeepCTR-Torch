# FAQ

## 1. Save or load weights/models
----------------------------------------
To save/load weights:

```python
import torch
model = DeepFM(...)
torch.save(model.state_dict(), 'DeepFM_weights.h5')
model.load_state_dict(torch.load('DeepFM_weights.h5'))
```

To save/load models:

```python
import torch
model = DeepFM(...)
torch.save(model, 'DeepFM.h5')
model = torch.load('DeepFM.h5')
```

## 2. Set learning rate and use earlystopping
---------------------------------------------------
Here is a example of how to set learning rate and earlystopping:

```python
from torch.optim import Adagrad
from deepctr_torch.models import DeepFM
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

model = DeepFM(linear_feature_columns,dnn_feature_columns)
model.compile(Adagrad(model.parameters(),0.1024),'binary_crossentropy',metrics=['binary_crossentropy'])

es = EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, verbose=1, patience=0, mode='min')
mdckpt = ModelCheckpoint(filepath='model.ckpt', monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')
history = model.fit(model_input,data[target].values,batch_size=256,epochs=10,verbose=2,validation_split=0.2,callbacks=[es,mdckpt])
print(history)
```

## 3. How to add a long dense feature vector as a input to the model?
```python
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import DenseFeat,SparseFeat,get_feature_names
import numpy as np

feature_columns = [SparseFeat('user_id',120,),SparseFeat('item_id',60,),DenseFeat("pic_vec",5)]
fixlen_feature_names = get_feature_names(feature_columns)

user_id = np.array([[1],[0],[1]])
item_id = np.array([[30],[20],[10]])
pic_vec = np.array([[0.1,0.5,0.4,0.3,0.2],[0.1,0.5,0.4,0.3,0.2],[0.1,0.5,0.4,0.3,0.2]])
label = np.array([1,0,1])

model_input = {'user_id':user_id,'item_id':item_id,'pic_vec':pic_vec}

model = DeepFM(feature_columns,feature_columns)
model.compile('adagrad','binary_crossentropy')
model.fit(model_input,label)
```

## 4. How to run the demo with GPU ?

```python
import torch
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = DeepFM(...,device=device)
```

## 5. How to run the demo with multiple GPUs ?

```python
model = DeepFM(..., device=device, gpus=[0, 1])
```

## 6. How to set different loss weights for the tasks of a multi-task model ?

Pass `loss_weights` to `compile` together with one loss per task. It can be a list in the order of `task_names` or a
dict keyed by task name. The training loss is `sum(loss_weights[t] * task_loss[t]) + regularization loss + auxiliary loss`.

```python
model = MMOE(dnn_feature_columns, task_types=['binary', 'regression'], task_names=['like', 'watch'])
model.compile("adam", loss=["binary_crossentropy", "mse"], loss_weights={"like": 1.0, "watch": 0.2})
```

- `loss_weights=None` (default) keeps the previous behavior: every task has weight 1.
- Weights are used as given (they are not normalized) and must be finite, non-negative and not all zero.
- A weight of 0 removes the data loss of that task. Its parameters can still change through layers shared with other
  tasks, regularization or weight decay.
