use candle_core::{Device, Result, Tensor};
use candle_nn as nn;
use nn::PReLU;

#[derive(Debug)]
enum LayerType {
    Conv2d(nn::Conv2d),
    PReLU(nn::PReLU),
}

#[derive(Debug)]
struct Sequential {
    modules: Vec<LayerType>,
}

impl Sequential {
    fn new(modules: Vec<LayerType>) -> Self {
        Self { modules }
    }

    fn add(&mut self, module: LayerType) {
        self.modules.push(module);
    }

    fn add_all(&mut self, modules: Vec<LayerType>) {
        self.modules.extend(modules);
    }
}

impl nn::Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out = xs.clone();
        for module in &self.modules {
            match module {
                LayerType::Conv2d(conv) => out = conv.forward(&out)?,
                LayerType::PReLU(act) => out = act.forward(&out)?,
            }
        }
        Ok(out)
    }
}

// pub enum ActType {
//     PReLU,
//     LeakyReLU,
//     ReLU,
// }

pub struct SRVGGNetCompact {
    body: Sequential,
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
        let mut body = Sequential {
            modules: Vec::with_capacity(2 * num_conv + 3),
        };
        body.add(LayerType::Conv2d(nn::conv2d(
            num_in_ch,
            num_feat,
            3,
            config,
            vb.pp("body.0"),
        )?));
        body.add(LayerType::PReLU(nn::prelu(
            Option::from(num_feat),
            vb.pp("body.1"),
        )?));
        // match act_type {
        //     ActType::PReLU => body.add(nn::prelu(Option::from(num_feat), vb.pp("body.1"))?),
        //     ActType::LeakyReLU => body.add(nn::Activation::LeakyRelu(0.1)),
        //     ActType::ReLU => body.add(nn::Activation::Relu),
        // };
        for i in 0..num_conv {
            body.add(LayerType::Conv2d(nn::conv2d(
                num_feat,
                num_feat,
                3,
                config,
                vb.pp(&format!("body.{}", 2 * i + 2)),
            )?));
            // match act_type {
            // ActType::PReLU =>
            body.add(LayerType::PReLU(nn::prelu(
                Option::from(num_feat),
                vb.pp(&format!("body.{}", 2 * i + 3)),
            )?));
            // _ => {}
            // ActType::LeakyReLU => body.add(nn::Activation::LeakyRelu(0.1)),
            // ActType::ReLU => body.add(nn::Activation::Relu),
            // };
        }
        body.add(LayerType::Conv2d(nn::conv2d(
            num_feat,
            num_out_ch * upscale * upscale,
            3,
            config,
            vb.pp(&format!("body.{}", 2 * num_conv + 2)),
        )?));
        Ok(Self { body, upscale })
    }
}

impl nn::Module for SRVGGNetCompact {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b_size, _channels, h, w) = xs.dims4()?;
        let out = self.body.forward(xs)?;
        let out = nn::ops::pixel_shuffle(&out, self.upscale)?;
        let base = xs.upsample_nearest2d(self.upscale * h, self.upscale * w)?;
        let out = (base + out)?;
        Ok(out)
    }
}
