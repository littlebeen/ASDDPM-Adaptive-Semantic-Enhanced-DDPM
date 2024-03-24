# ASDDPM-Adaptive-Semantic-Enhanced-DDPM
The code for [Adaptive Semantic-Enhanced Denoising Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution](https://arxiv.org/abs/2403.11078)

**Models**

* Dit: [Scalable Diffusion Models with Transformers](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)

* SRDiff:[SRDiff: Single image super-resolution with diffusion probabilistic models](https://pdf.sciencedirectassets.com/271597/1-s2.0-S0925231222X00052/1-s2.0-S0925231222000522/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFcaCXVzLWVhc3QtMSJGMEQCIE5nQ5z%2FbWYGT5VLwQVLg8vMxQxFNx2MYuYzTVERBRExAiAlCDT9ls%2BtTSxeJcfLejFw0N39X8cpQBsLUFB%2BH%2F3pViqyBQhwEAUaDDA1OTAwMzU0Njg2NSIMOoZ3B59EyhVGS5wfKo8FGT1xBXoR9PjQ6DBggoRXUfIg3L1rAqLSpc9EsLFEvPqs3PVcpVJ%2BnCq%2F556EbYRhzRY%2BVuAxyhksJ8MuENXJlspwgo5QFxVQI6%2F7wCknpgvF8SbE6me7sYOS14Z07XwKIKQmHh5%2FXk%2FlY9XHL4QxUItcQoBeaPLp%2BCSX%2FVdZJHFQMCzz2hi5kIE1H7CMgbW1QWil6cAUPJubJ6VCq2yVLU3WeYXTDbysPi0rAfA5o%2FQ1fDfPjhBfLs%2BLotkqVWZwsVpROvx%2BODikwIZOBqbZQVKAuFtwNEaMsbIPiI5GJKiCw9%2BZIKUDABKYIUjhw59VKI0z5hcIXzzvcEAPlYHp4b40AZzW7CmMRPtC4yEgoo37vPW9%2FGoagcO41HTn4kiKYoDNkRAzu06r36mFvmMmoQhGYBLmQZslRzK1h2pCb24qXFYQYMSNTcT637E8WToHZSIB86LmLMMTIXTB79%2F2pIhBv2hvCy86tGSfU5mjwCVwxp80GFLVYuj3r7kOhSE4Mc8pcsGg%2FulwjcrDfCGaQ1IDJ4pOk0qZHxd3urYmDbSlrv6PVwPzbECd2%2FFRFlZ5%2BdUac3bhRRyEyWS6J1%2B6qgsvkxaG3n6YH%2Bp2i1xalICFUL9WhphkJH%2BY%2FjBdJ%2FhrGiIfuq%2BG7iQG6MBvZaUN9ecQ8y9DNUVIAqKkzjrqGVyn%2B6%2Fff9mYMyQQDDtsgfTSx51sZzcUlAWUcKAn6%2F867ObBOTh5nbY1kYR3Hi88A9%2BLw4TfZN8i8VozSVtMTTa9ejBj9XxZE8HphhfVwPR4tMKIb99I0ABgjq28mp3zTf1mInAHFGdmQRpzkhUmIqb%2BcCcrm6idYkWvisgulhiyK2TZTrXBqUssCgaWa%2FnYYzDXnP%2BvBjqyATy0kHH0P%2FNl2sBQyMouHkX1TR2ZmMGFv%2Fo4fEu2neZMKDGBtLczPaqrLFLlpxG8LixyEuoE3UQ307bO3orpabfFwomQ7dDYDawtxDgVJpnbQx%2BcGjvtayLMQtOJXhhLfbE2Bei2tnpBBjzWO1KY9G7GGSMnFEyVLZIXzzZtRuT%2FZEMJgESWSA2QxiIxec8oeRtHD2wrEbsqKlX38a2wahJ1S5dp0zpd2ebdqY1JswQ%2BYnU%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240324T072906Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYV4XKUYVT%2F20240324%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7a90f3f3e6bcd0fefa0542db9777c5196389f36bcef7927f35dd71ca05249f45&hash=2d9b7c74023e4338242573ec8c7e752c9eaf87dfda1b5195f9402d3056a731a7&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0925231222000522&tid=spdf-b0bc34aa-690e-41e1-91f5-38b84a8007af&sid=28a2b10537db2743d82a1a826a2993ab5976gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=190a5c535a53005d0004&rr=8694f2bccea917e6&cc=cn)

* ASDDPM: [Adaptive Semantic-Enhanced Denoising Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution](https://arxiv.org/abs/2403.11078)

**Dataset**

The experimental datasets, OLI2MSI and Alsat, could be obtained from:

* [OLI2MSI](https://github.com/wjwjww/OLI2MSI)

* [Alsat](https://github.com/achrafdjerida/Alsat-2B)

# Usage

**Train**

1. Change the model name and data information in the option.py

```python src/main.py ```

**Test**

1. Put pre-trained model into 'pre_train'
2. Change the model name in the option.py

```python test.py```

**Weight**

Change the github branch to 'weight' and download.

# Cite

```
@article{sui2024adaptive,
  title={Adaptive Semantic-Enhanced Denoising Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution},
  author={Sui, Jialu and Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={arXiv preprint arXiv:2403.11078},
  year={2024}
}
@article{sui2023gcrdn,
  title={Gcrdn: Global context-driven residual dense network for remote sensing image super-resolution},
  author={Sui, Jialu and Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```
