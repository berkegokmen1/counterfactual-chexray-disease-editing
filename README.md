# Counterfactual Disease Removal and Generation in Chest X-Rays Using Diffusion Models

[Paper](https://berkegokmen1.github.io/counterfactual-disease-removal-and-generation-chest-xray/) | [Project Website](https://berkegokmen1.github.io/counterfactual-disease-removal-and-generation-chest-xray/) | [BibTeX](#bibtex)

## Authors
[Ahmet Berke Gökmen](https://berkegokmen1.github.io/), [Ender Konukoglu](https://people.ee.ethz.ch/~kender/)

![teaser](https://github.com/user-attachments/assets/4faf0674-66e3-45e7-bb56-c2c2caeb6ab1)

## TODO
- [X] Release Website
- [X] Release Code
- [ ] Run Instructions

## Setup

```bash
conda create -n chexray-editing python=3.10
pip install -r requirements.txt [TODO]
```

## Inference
Please download chexzero, chexpert and chexray-diffusion checkpoints from their respective repositories and update the paths in `config.yaml`.

In additon to the checkpoints, you'll need to download `CheXpert-v1.0-small` dataset from the official chexpert website or you may use any chest x-ray image.
```bash
python finetune_sample.py --config config.yaml --target "Pleural Effusion" --mode "removal" --experiment_name "demo"
```

## Questions

You may reach me through [LinkedIn](https://www.linkedin.com/in/berkegokmen/).

## This work would not have been possible without:
- https://github.com/rajpurkarlab/CheXzero
- https://github.com/jfhealthcare/Chexpert
- https://github.com/saiboxx/chexray-diffusion

## BibTeX
```
@misc{,
      title={Counterfactual Disease Removal and Generation in Chest X-Rays Using Diffusion Models}, 
      author={Ahmet Berke Gokmen and Ender Konukoglu},
      year={2024},
      eprint={},
      archivePrefix={},
      primaryClass={cs.CV},
      url={}, 
}
```

<hr>

<div align="center">
  <img src="https://profile-counter.glitch.me/counterfactual-chexray-disease-editing/count.svg"  />
</div>

