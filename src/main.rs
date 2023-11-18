use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
mod new_arch;
mod old_arch;
use image::DynamicImage;
use image::RgbImage;
use new_arch::RRDBNet as RealESRGAN;
use old_arch::RRDBNet as OldESRGAN;

fn img2tensor(img: DynamicImage, device: &Device) -> Tensor {
    let height: usize = img.height() as usize;
    let width: usize = img.width() as usize;
    let data = img.to_rgb8().into_raw();
    let tensor = Tensor::from_vec(data, (height, width, 3), device)
        .unwrap()
        .permute((2, 0, 1))
        .unwrap();
    let image_t = (tensor.unsqueeze(0).unwrap().to_dtype(DType::F32).unwrap() / 255.).unwrap();
    return image_t;
}

fn tensor2img(tensor: Tensor) -> RgbImage {
    let cpu = Device::Cpu;

    let result = tensor
        .permute((1, 2, 0))
        .unwrap()
        .detach()
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_dtype(DType::U8)
        .unwrap();

    let dims = result.dims();
    let height = dims[0];
    let width = dims[1];

    let data = result.flatten_to(2).unwrap().to_vec1::<u8>().unwrap();
    let out_img = RgbImage::from_vec(width as u32, height as u32, data).unwrap();
    out_img
}

fn process(model: &RealESRGAN, img: DynamicImage, device: &Device) -> RgbImage {
    let img_t = img2tensor(img, &device);

    let now = Instant::now();
    let result = model.forward(&img_t).unwrap();
    println!("Model took {:?}", now.elapsed());

    let result = (result.squeeze(0).unwrap().clamp(0., 1.).unwrap() * 255.).unwrap();

    let out_img = tensor2img(result);
    return out_img;
}

fn main() {
    let device = Device::new_cuda(0).unwrap();

    let model_path = "model.safetensors";

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device).unwrap() };

    let model = RealESRGAN::load(vb, 3, 3, 4, 64, 23, 32).unwrap();

    let images_dir = r"path/to/images";
    let out_dir = r"./rust-out";

    if std::path::Path::new(out_dir).exists() {
        std::fs::remove_dir_all(out_dir).unwrap();
    }
    std::fs::create_dir(out_dir).unwrap();

    let files = std::fs::read_dir(images_dir).unwrap();
    let now = Instant::now();
    files.into_iter().for_each(|file| {
        let file = file.unwrap();
        let path = file.path();
        let img = image::open(path).unwrap();

        let out_img = process(&model, img, &device);

        let out_path = format!("{}/{}", out_dir, file.file_name().into_string().unwrap());
        out_img.save(out_path).unwrap();
        println!("Saved {}", file.file_name().into_string().unwrap());
    });
    println!("Time taken: {:?}", now.elapsed());
}
