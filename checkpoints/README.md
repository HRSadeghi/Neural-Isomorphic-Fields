# checkpoints

The model weights are not in git. Download them from
[Google Drive](https://drive.google.com/drive/folders/1jSvimFyft3ftIpodHPicKbEaBF4jsdvn).

After unpacking, the folder looks like this:

```
checkpoints/
  main_s0/best.pt  main_s1/best.pt  main_s2/best.pt
  full_*/best.pt  aeonly/best.pt  aeonly_rec/best.pt  emb_c*/best.pt  len_n*_c*/best.pt
  abl_*/best.pt  depth_*/best.pt
  language_model/operator_layers_6000.pt
  language_model/operator_layers_10000.pt
  SHA256SUMS
```

* `<run>/best.pt` (43 files, 2.2 GB in total): the checkpoint of the run with the best validation score, as a dict with
  `model` (state dict), `cfg` (arguments of `nif.model.NIF`), `step` and `score`. Load it with `evaluate.load(path)`.
  The operator runs (`abl_*`, `depth_*`) store the weights of `train_operator.py` instead.
* `language_model/operator_layers_6000.pt` and `operator_layers_10000.pt` (2.8 GB each): the weights of the
  language model with the NIF operator layers after 6000 and 10,000 steps (`state`, `step`). The NIF part is not
  included; it is read from `main_s0/best.pt`.

To check the files after downloading:

```bash
cd checkpoints && sha256sum -c SHA256SUMS                  # Linux
cd checkpoints && shasum -a 256 -c SHA256SUMS              # macOS
```

```powershell
Get-FileHash checkpoints\main_s0\best.pt -Algorithm SHA256  # Windows, one file; compare with SHA256SUMS
```

Scripts look for the weights of a run first in `runs/<run>/best.pt` (after training it yourself) and then here.
