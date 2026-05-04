# Differential Surfel Rasterization

This is the differentiable rasterization engine used in [ReLaGS: Relational Language Gaussian Splatting](https://github.com/dfki-av/ReLaGS) (CVPR 2025).

It is built upon [2D Gaussian Splatting (2DGS)](https://github.com/hbb1/2d-gaussian-splatting) and extends it with ID-based rendering for semantic feature tracking and segmentation.


If you find this repository useful, please consider citing:

> ReLaGS:

```bibTeX
@inproceedings{xiearafa2026relags,
  title     = {ReLaGS: Relational Language Gaussian Splatting},
  author    = {Xie, Yaxu and Arafa, Abdalla and Javanmardi, Alireza and Millerdurai, Christen and Hu, Jia Cheng and Wang, Shaoxiang and Pagani, Alain and Stricker, Didier},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

> Along with 2DGS and 3DGS:

```bibTeX
@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```

```bibTeX
@article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {ACM Transactions on Graphics},
    number       = {4},
    volume       = {42},
    month        = {July},
    year         = {2023},
    url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```