# Differential Surfel Rasterization

This is the differentiable rasterization engine used in [ReLaGS: Relational Language Gaussian Splatting](https://github.com/dfki-av/ReLaGS) (CVPR 2025).

It is built upon [2D Gaussian Splatting (2DGS)](https://github.com/hbb1/2d-gaussian-splatting) and extends it with ray-tracing and max-contribution rendering for semantic feature aggregation.


If you find this useful, please consider citing:

> ReLaGS:

```bibTeX
@inproceedings{xiearafa2026relags,
  title     = {ReLaGS: Relational Language Gaussian Splatting},
  author    = {Xie, Yaxu and Arafa, Abdalla and Javanmardi, Alireza and Millerdurai, Christen and Hu, Jia Cheng and Wang, Shaoxiang and Pagani, Alain and Stricker, Didier},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

> Along with 2DGS and THGS:

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
@article{thgs2025,
    title={Training-Free Hierarchical Scene Understanding for Gaussian Splatting with Superpoint Graphs},
    author={Dai, Shaohui and Qu, Yansong and Li, Zheyan and Li, Xinyang and Zhang, Shengchuan and Cao, Liujuan},
    journal={arXiv preprint arXiv:2504.13153},
    year={2025}
}
```