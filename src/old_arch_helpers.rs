use std::collections::{HashMap, HashSet};

use candle_core::Tensor;
use std::format;

pub fn get_scale(state_dict: &HashMap<String, Tensor>) -> usize {
    // The idea is pretty simple: model.0 and model.1 are always present
    // At the end, there are always two more layers.
    // Therefore, the number of upsample layers is the total number of unique model.#.* keys (with unique numbers) minus 4
    // Then, we just need to do 2**num_upsample_layers to get the scale
    let num_unique_layers = state_dict
        .keys()
        .filter(|x| x.contains("model."))
        .map(|x| x.split(".").collect::<Vec<_>>()[1])
        .collect::<HashSet<_>>()
        .len();
    return usize::pow(2, (num_unique_layers - 4) as u32);
}

pub fn get_in_nc(state_dict: &HashMap<String, Tensor>) -> usize {
    return match state_dict.get("model.0.weight") {
        Some(x) => x.shape().dims()[1],
        None => 3,
    };
}

pub fn get_out_nc(state_dict: &HashMap<String, Tensor>) -> usize {
    let highest_layer_num = state_dict
        .keys()
        .filter(|x| x.contains("model."))
        .map(|x| x.split(".").collect::<Vec<&str>>()[1])
        .map(|y| y.parse::<usize>().unwrap())
        .max()
        .unwrap();
    return state_dict
        .get(&format!("model.{highest_layer_num}.weight"))
        .unwrap()
        .shape()
        .dims()[0];
}
