import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base.modules import Conv2dReLU

# isort: off
from spacenet8_model.utils.misc import get_flatten_classes
# isort: on


class SiameseModel(torch.nn.Module):
    def __init__(self, config, **kwargs):
        assert config.Model.n_input_post_images in [1, 2], config.Model.n_input_post_images

        super().__init__()

        # siamese branch
        n_classes = len(get_flatten_classes(config))
        self.branch = smp.create_model(
            config.Model.arch,
            encoder_name=config.Model.encoder,
            in_channels=3,
            classes=n_classes,
            encoder_weights="imagenet",
            **kwargs)

        branch_out_channels = self.branch.segmentation_head[0].in_channels
        self.branch.segmentation_head[0] = torch.nn.Identity()

        # post head
        if config.Model.enable_siamese_post_head:
            kernel_size = config.Model.post_head_kernel_size
            padding = (kernel_size - 1) // 2
            if config.Model.post_head_module == 'conv':
                self.post_head = [torch.nn.Conv2d(
                    branch_out_channels,
                    branch_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding) for _ in range(config.Model.n_post_head_modules)
                ]
                self.post_head = torch.nn.Sequential(*self.post_head)
            elif config.Model.post_head_module in ['conv_relu', 'conv_bn_relu']:
                use_batchnorm = config.Model.post_head_module == 'conv_bn_relu'
                self.post_head = [Conv2dReLU(
                    branch_out_channels,
                    branch_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    use_batchnorm=use_batchnorm) for _ in range(config.Model.n_post_head_modules)
                ]
                self.post_head = torch.nn.Sequential(*self.post_head)
            elif config.Model.post_head_module == 'average_pool':
                self.post_head = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
            elif config.Model.post_head_module == 'max_pool':
                self.post_head = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
            else:
                raise ValueError(config.Model.n_post_head_modules)
        else:
            self.post_head = torch.nn.Identity()

        # siamese head
        head_in_channels = branch_out_channels * (1 + config.Model.n_input_post_images)
        kernel_size = config.Model.siamese_head_kernel_size
        padding = (kernel_size - 1) // 2
        assert config.Model.n_siamese_head_convs >= 1

        head_module = config.Model.siamese_head_module
        if head_module == 'conv':
            head = [torch.nn.Conv2d(
                        head_in_channels,
                        head_in_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding) for _ in range(config.Model.n_siamese_head_convs - 1)
                    ]
        elif head_module in ['conv_relu', 'conv_bn_relu']:
            use_batchnorm = head_module == 'conv_bn_relu'
            head = [Conv2dReLU(
                        head_in_channels,
                        head_in_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        use_batchnorm=use_batchnorm) for _ in range(config.Model.n_siamese_head_convs - 1)
                    ]
        else:
            raise ValueError(head_module)

        kernel_size = config.Model.siamese_head_kernel_size_last
        dilation = config.Model.siamese_head_dilation_last
        padding = dilation * ((kernel_size - 1) // 2)
        head.append(
            torch.nn.Conv2d(
                head_in_channels,
                n_classes,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation
            )
        )
        self.head = torch.nn.Sequential(*head)

        self.n_input_post_images = config.Model.n_input_post_images

    def forward(self, image, images_post):
        assert len(images_post) == self.n_input_post_images

        x = [self.branch(image)]
        for i in range(self.n_input_post_images):
            x.append(self.post_head(self.branch(images_post[i])))
        x = torch.cat(x, dim=1)
        return self.head(x)
