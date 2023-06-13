This README explains how I prepared XD_XD's SpaceNet-5 winning models 
(pre-trained weights uploaded under s3://spacenet-dataset/spacenet-model-weights/spacenet-5/xd_xd).

XD_XD's models are [not saved as PyTorch state_dict](https://github.com/SpaceNetChallenge/SpaceNet_Optimized_Routing_Solutions/blob/3fbc215de6b05904a5b54b2c7cde7e61074ae38d/xd_xd/aa/pytorch/callbacks/__init__.py#L79-L81),
and this causes PyTorch version error (more specifically, `torch.nn.Upsample` error) when calling `forward()` of the model.

So I saved these models as PyTorch state_dict and then upload them to AWS S3 bucket.

## Procedure

Download XD-XD's models from SpaceNet S3:

```
aws s3 cp --recursive s3://spacenet-dataset/spacenet-model-weights/spacenet-5/xd_xd /path/to/xdxd_models
```

Download XD_XD's code and copy `aa/` folder under project root.

```
git clone https://github.com/SpaceNetChallenge/SpaceNet_Optimized_Routing_Solutions.git
cp -r SpaceNet_Optimized_Routing_Solutions/xd_xd/aa/ /work
rm -rf SpaceNet_Optimized_Routing_Solutions/
```

Convert them into PyTorch state_dict:

```
python tools/prepare_xdxd_sn5_models.py -i /path/to/xdxd_models -o /path/to/state_dict
```

I uploaded converted state_dict to AWS S3, which are used in the final scoring.
