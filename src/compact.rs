use candle_core::{Result, Tensor};
use candle_nn as nn;

// pub enum ActType {
//     PReLU,
//     LeakyReLU,
//     ReLU,
// }

pub struct SRVGGNetCompact {
    body: nn::Sequential,
    upscale: usize,
}

impl SRVGGNetCompact {
    pub fn load(
        vb: nn::VarBuilder,
        num_in_ch: usize,
        num_out_ch: usize,
        num_feat: usize,
        num_conv: usize,
        upscale: usize,
        // act_type: ActType,
    ) -> Result<Self> {
        let config = nn::Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let mut body = nn::seq();
        // let mut body = Sequential {
        //     modules: Vec::with_capacity(2 * num_conv + 3),
        // };
        body = body.add(nn::conv2d(num_in_ch, num_feat, 3, config, vb.pp("body.0"))?);
        body = body.add(nn::prelu(Option::from(num_feat), vb.pp("body.1"))?);
        // match act_type {
        //     ActType::PReLU => body.add(nn::prelu(Option::from(num_feat), vb.pp("body.1"))?),
        //     ActType::LeakyReLU => body.add(nn::Activation::LeakyRelu(0.1)),
        //     ActType::ReLU => body.add(nn::Activation::Relu),
        // };
        for i in 0..num_conv {
            body = body.add(nn::conv2d(
                num_feat,
                num_feat,
                3,
                config,
                vb.pp(&format!("body.{}", 2 * i + 2)),
            )?);
            // match act_type {
            // ActType::PReLU =>
            body = body.add(nn::prelu(
                Option::from(num_feat),
                vb.pp(&format!("body.{}", 2 * i + 3)),
            )?);
            // _ => {}
            // ActType::LeakyReLU => body.add(nn::Activation::LeakyRelu(0.1)),
            // ActType::ReLU => body.add(nn::Activation::Relu),
            // };
        }
        let body = body.add(nn::conv2d(
            num_feat,
            num_out_ch * upscale * upscale,
            3,
            config,
            vb.pp(&format!("body.{}", 2 * num_conv + 2)),
        )?);
        Ok(Self { body, upscale })
    }
}

impl nn::Module for SRVGGNetCompact {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b_size, _channels, h, w) = xs.dims4()?;
        let base = xs.upsample_nearest2d(self.upscale * h, self.upscale * w)?;
        let out = self.body.forward(xs)?;
        let out = nn::ops::pixel_shuffle(&out, self.upscale)?;
        let out = (base + out)?;
        Ok(out)
    }
}
