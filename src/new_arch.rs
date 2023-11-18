use candle_core::{Result, Tensor};
use candle_nn as nn;

#[derive(Debug)]
struct Upsample {
    scale_factor: usize,
}

impl Upsample {
    fn new(scale_factor: usize) -> Result<Self> {
        Ok(Upsample { scale_factor })
    }
}

impl nn::Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b_size, _channels, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(self.scale_factor * h, self.scale_factor * w)
    }
}

#[derive(Debug)]
struct ResidualDenseBlock {
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
    conv3: nn::Conv2d,
    conv4: nn::Conv2d,
    conv5: nn::Conv2d,
    lrelu: nn::Activation,
}

impl ResidualDenseBlock {
    fn load(vb: nn::VarBuilder, num_feat: usize, num_grow_ch: usize) -> Result<Self> {
        let config = nn::Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let conv1 = nn::conv2d(num_feat, num_grow_ch, 3, config, vb.pp("conv1"));
        let conv2 = nn::conv2d(
            num_feat + num_grow_ch,
            num_grow_ch,
            3,
            config,
            vb.pp("conv2"),
        );
        let conv3 = nn::conv2d(
            num_feat + 2 * num_grow_ch,
            num_grow_ch,
            3,
            config,
            vb.pp("conv3"),
        );
        let conv4 = nn::conv2d(
            num_feat + 3 * num_grow_ch,
            num_grow_ch,
            3,
            config,
            vb.pp("conv4"),
        );
        let conv5 = nn::conv2d(
            num_feat + 4 * num_grow_ch,
            num_feat,
            3,
            config,
            vb.pp("conv5"),
        );
        let lrelu = nn::Activation::LeakyRelu(0.2);
        Ok(Self {
            conv1: conv1.unwrap(),
            conv2: conv2.unwrap(),
            conv3: conv3.unwrap(),
            conv4: conv4.unwrap(),
            conv5: conv5.unwrap(),
            lrelu,
        })
    }
}

impl nn::Module for ResidualDenseBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x1 = self.lrelu.forward(&self.conv1.forward(xs)?)?;
        let x2 = self
            .lrelu
            .forward(&self.conv2.forward(&Tensor::cat(&[xs, &x1], 1)?)?)?;
        let x3 = self
            .lrelu
            .forward(&self.conv3.forward(&Tensor::cat(&[xs, &x1, &x2], 1)?)?)?;
        let x4 = self
            .lrelu
            .forward(&self.conv4.forward(&Tensor::cat(&[xs, &x1, &x2, &x3], 1)?)?)?;
        let x5 = self
            .conv5
            .forward(&Tensor::cat(&[xs, &x1, &x2, &x3, &x4], 1)?)?;
        Ok((x5 * 0.2 + xs)?)
    }
}

#[derive(Debug)]
struct RRDB {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl RRDB {
    fn load(vb: nn::VarBuilder, num_feat: usize, num_grow_ch: usize) -> Result<Self> {
        let rdb1 = ResidualDenseBlock::load(vb.pp("rdb1"), num_feat, num_grow_ch);
        let rdb2 = ResidualDenseBlock::load(vb.pp("rdb2"), num_feat, num_grow_ch);
        let rdb3 = ResidualDenseBlock::load(vb.pp("rdb3"), num_feat, num_grow_ch);
        Ok(Self {
            rdb1: rdb1.unwrap(),
            rdb2: rdb2.unwrap(),
            rdb3: rdb3.unwrap(),
        })
    }
}

impl nn::Module for RRDB {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x1 = self.rdb1.forward(xs)?;
        let x2 = self.rdb2.forward(&x1)?;
        let x3 = self.rdb3.forward(&x2)?;
        Ok((x3 * 0.2 + xs)?)
    }
}

#[derive(Debug)]
struct Sequential {
    modules: Vec<RRDB>,
}

impl Sequential {
    fn new(modules: Vec<RRDB>) -> Self {
        Self { modules }
    }
}

impl nn::Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out = xs.clone();
        for module in &self.modules {
            out = module.forward(&out)?;
        }
        Ok(out)
    }
}

#[derive(Debug)]
pub struct RRDBNet {
    conv_first: nn::Conv2d,
    body: Sequential,
    conv_body: nn::Conv2d,
    conv_up1: nn::Conv2d,
    conv_up2: nn::Conv2d,
    conv_hr: nn::Conv2d,
    conv_last: nn::Conv2d,
    lrelu: nn::Activation,
}

impl RRDBNet {
    pub fn load(
        vb: nn::VarBuilder,
        num_in_ch: usize,
        num_out_ch: usize,
        scale: usize,
        num_feat: usize,
        num_blocks: usize,
        num_grow_ch: usize,
    ) -> Result<Self> {
        let config = nn::Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let conv_first = nn::conv2d(num_in_ch, num_feat, 3, config, vb.pp("conv_first"));
        let body = Sequential {
            modules: (0..num_blocks)
                .map(|i| RRDB::load(vb.pp(format!("body.{i}")), num_feat, num_grow_ch))
                .collect::<Result<Vec<_>>>()?,
        };
        let conv_body = nn::conv2d(num_feat, num_feat, 3, config, vb.pp("conv_body"));
        let conv_up1 = nn::conv2d(num_feat, num_feat, 3, config, vb.pp("conv_up1"));
        let conv_up2 = nn::conv2d(num_feat, num_feat, 3, config, vb.pp("conv_up2"));
        let conv_hr = nn::conv2d(num_feat, num_feat, 3, config, vb.pp("conv_hr"));
        let conv_last = nn::conv2d(num_feat, num_out_ch, 3, config, vb.pp("conv_last"));
        let lrelu = nn::Activation::LeakyRelu(0.2);
        Ok(Self {
            conv_first: conv_first.unwrap(),
            body,
            conv_body: conv_body.unwrap(),
            conv_up1: conv_up1.unwrap(),
            conv_up2: conv_up2.unwrap(),
            conv_hr: conv_hr.unwrap(),
            conv_last: conv_last.unwrap(),
            lrelu,
        })
    }
}

impl nn::Module for RRDBNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut feat = self.conv_first.forward(xs)?;
        let body_feat = self.conv_body.forward(&self.body.forward(&feat)?)?;
        feat = (feat + body_feat)?;
        feat = self
            .lrelu
            .forward(&self.conv_up1.forward(&Upsample::new(2)?.forward(&feat)?)?)?;
        feat = self
            .lrelu
            .forward(&self.conv_up2.forward(&Upsample::new(2)?.forward(&feat)?)?)?;
        let out = self
            .conv_last
            .forward(&self.lrelu.forward(&self.conv_hr.forward(&feat)?)?)?;
        Ok(out)
    }
}
