use std::time::Instant;

use candle_core::{pickle, DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
mod new_arch;
mod old_arch;
use candle_core::safetensors::load;
use clap::ValueEnum;
use compact::SRVGGNetCompact as Compact;
use image::DynamicImage;
use image::RgbImage;
use new_arch::RRDBNet as RealESRGAN;
use old_arch::RRDBNet as OldESRGAN;
use std::path::Path;
mod old_arch_helpers;
use old_arch_helpers::{get_in_nc, get_nb, get_nf, get_out_nc, get_scale};
mod compact;

use clap::Parser;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ModelType {
    /// Old-arch ESRGAN
    Old,
    /// New-arch ESRGAN (RealESRGAN)
    New,
    // RealESRGANv2 aka Compact
    Compact,
}

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model file in safetensors format
    #[arg(short, long)]
    model: String,

    /// Folder path containing images to upscale
    #[arg(short, long)]
    input: String,

    /// Folder path to save upscaled images
    #[arg(short, long)]
    output: String,

    /// Device to run the model on
    /// -1 for CPU, 0 for GPU 0, 1 for GPU 1, etc.
    #[arg(short, long, default_value = "-1")]
    device: i32,

    /// Architecture revision (old or new). Dependent on the model used.
    #[arg(short, long, value_enum)]
    arch: Option<ModelType>,

    /// Number of input channels. Dependent on the model used.
    #[arg(long)]
    in_channels: Option<usize>,

    /// Number of output channels. Dependent on the model used.
    #[arg(long)]
    out_channels: Option<usize>,

    /// Number of RRDB blocks. Dependent on the model used.
    #[arg(long)]
    num_blocks: Option<usize>,

    /// Number of features. Dependent on the model used.
    #[arg(long)]
    num_features: Option<usize>,

    /// Scale of the model. Dependent on the model used.
    #[arg(short, long)]
    scale: Option<usize>,

    /// Run the model with half precision (fp16)
    #[arg(long)]
    half: bool,
}

fn img2tensor(img: DynamicImage, device: &Device, half: bool) -> Tensor {
    let height: usize = img.height() as usize;
    let width: usize = img.width() as usize;
    let data = img.to_rgb8().into_raw();
    let tensor = Tensor::from_vec(data, (height, width, 3), device)
        .unwrap()
        .permute((2, 0, 1))
        .unwrap();
    let image_t = (tensor
        .unsqueeze(0)
        .unwrap()
        .to_dtype(if half { DType::F16 } else { DType::F32 })
        .unwrap()
        / 255.)
        .unwrap();
    return image_t;
}

fn tensor2img(tensor: Tensor) -> RgbImage {
    let cpu = Device::Cpu;

    let result = tensor
        .permute((1, 2, 0))
        .unwrap()
        .detach()
        .unwrap()
        .to_dtype(DType::U8)
        .unwrap()
        .to_device(&cpu)
        .unwrap();

    let dims = result.dims();
    let height = dims[0];
    let width = dims[1];

    let data = result.flatten_to(2).unwrap().to_vec1::<u8>().unwrap();
    let out_img = RgbImage::from_vec(width as u32, height as u32, data).unwrap();
    out_img
}

enum ModelVariant {
    Old(OldESRGAN),
    New(RealESRGAN),
    Compact(Compact),
}

fn process(model: &ModelVariant, img: DynamicImage, device: &Device, half: bool) -> RgbImage {
    let img_t = img2tensor(img, &device, half);

    let now = Instant::now();
    let result = match model {
        ModelVariant::Old(model) => model.forward(&img_t).unwrap(),
        ModelVariant::New(model) => model.forward(&img_t).unwrap(),
        ModelVariant::Compact(model) => model.forward(&img_t).unwrap(),
    };
    println!("Model took {:?}", now.elapsed());

    let result = (result.squeeze(0).unwrap().clamp(0., 1.).unwrap() * 255.).unwrap();

    let out_img = tensor2img(result);
    return out_img;
}

fn main() {
    let args: Args = Args::parse();

    let device = match args.device {
        -1 => Device::Cpu,
        _ => Device::new_cuda(args.device as usize).unwrap(),
    };

    let path_extension = Path::new(&args.model)
        .extension()
        .unwrap()
        .to_str()
        .unwrap();

    let state_dict = match path_extension {
        "safetensors" => load(&args.model, &device).unwrap(),
        "pth" => pickle::read_all(&args.model).unwrap().into_iter().collect(),
        _ => panic!("Invalid model file extension"),
    };

    let vb = {
        VarBuilder::from_tensors(
            state_dict.clone(),
            if args.half { DType::F16 } else { DType::F32 },
            &device,
        )
    };

    let model_arch =
        args.arch
            .unwrap_or(if state_dict.keys().any(|x| x.contains("model.0.weight")) {
                ModelType::Old
            } else {
                ModelType::New
            });

    let model: ModelVariant = match model_arch {
        ModelType::Old => ModelVariant::Old(
            OldESRGAN::load(
                vb,
                args.in_channels.unwrap_or(get_in_nc(&state_dict)),
                args.out_channels.unwrap_or(get_out_nc(&state_dict)),
                args.scale.unwrap_or(get_scale(&state_dict)),
                args.num_features.unwrap_or(get_nf(&state_dict)),
                args.num_blocks.unwrap_or(get_nb(&state_dict)),
                32,
            )
            .unwrap(),
        ),
        ModelType::New => ModelVariant::New(
            RealESRGAN::load(
                vb,
                args.in_channels.unwrap_or(3),
                args.out_channels.unwrap_or(3),
                args.scale.unwrap_or(4),
                args.num_features.unwrap_or(64),
                args.num_blocks.unwrap_or(23),
                32,
            )
            .unwrap(),
        ),
        ModelType::Compact => ModelVariant::Compact(
            Compact::load(
                vb,
                args.in_channels.unwrap_or(3),
                args.out_channels.unwrap_or(3),
                args.num_features.unwrap_or(64),
                args.num_blocks.unwrap_or(23),
                args.scale.unwrap_or(4),
            )
            .unwrap(),
        ),
    };

    let images_dir = args.input;
    let out_dir = args.output;

    if !std::path::Path::new(&out_dir).exists() {
        std::fs::create_dir(&out_dir).unwrap();
    }

    let files = std::fs::read_dir(images_dir).unwrap();
    let now = Instant::now();
    files.into_iter().for_each(|file| {
        let file = file.unwrap();
        let path = file.path();
        let img = image::open(path).unwrap();

        let out_img = process(&model, img, &device, args.half);

        let out_path = format!("{}/{}", out_dir, file.file_name().into_string().unwrap());
        out_img.save(out_path).unwrap();
        println!("Saved {}", file.file_name().into_string().unwrap());
    });
    println!("Time taken: {:?}", now.elapsed());
}
